import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.BaseModel import BaseModel
from Utils_.utils import *
from random import randint
import scipy.sparse as sp
from models.BaseModel import BaseModel

class SelfGNN(BaseModel):
    reader = 'Handler'
    runner = 'Runner'
    
    def __init__(self, args, handler):
        super(SelfGNN, self).__init__(args, handler)
        self.args = args
        self.handler = handler
        self.regParams = list()

        self.prepareModel()

    def prepareModel(self):
        self.maxTime=self.handler.maxTime
        # self.keepRate = args.keepRate
        self.leaky = self.args.leaky
        self.actFunc = 'leakyRelu'
        
        self.uEmbed=nn.Parameter(torch.empty(self.args.graphNum, self.args.user, self.args.latdim), requires_grad=True)
        nn.init.xavier_uniform_(self.uEmbed)
        self.regParams.append(self.uEmbed)
        self.iEmbed=nn.Parameter(torch.empty(self.args.graphNum, self.args.item, self.args.latdim), requires_grad=True)
        nn.init.xavier_uniform_(self.iEmbed)
        self.regParams.append(self.iEmbed)
        self.posEmbed=nn.Parameter(torch.empty(self.args.pos_length, self.args.latdim), requires_grad=True)
        nn.init.xavier_uniform_(self.posEmbed)
        self.regParams.append(self.posEmbed)
        self.pos= torch.tile(torch.unsqueeze(torch.arange(self.args.pos_length),0),(self.args.batch,1))
        self.items=torch.arange(self.args.item)
        self.users=torch.arange(self.args.user)
        self.timeEmbed=nn.Parameter(torch.empty(self.maxTime+1, self.args.latdim), requires_grad=True)
        nn.init.xavier_uniform_(self.timeEmbed)
        self.regParams.append(self.timeEmbed)
        
        self.rnn = nn.LSTM(
            input_size=self.args.latdim, 
            hidden_size=self.args.latdim, 
            num_layers=1, 
            batch_first=True
        )
        self.rnn_dropout = nn.Dropout(p=1.0 - self.args.keepRate)
        
        # self.AdditiveAttention_0 = AdditiveAttention_Pytorch(self.args.query_vector_dim,self.args.latdim)
        # self.AdditiveAttention_1 = AdditiveAttention_Pytorch(self.args.query_vector_dim,self.args.latdim)
        
        self.MultiHeadSelfAttention_0 = MultiHeadSelfAttention_Pytorch(self.args.latdim,self.args.num_attention_heads)
        self.MultiHeadSelfAttention_1 = MultiHeadSelfAttention_Pytorch(self.args.latdim,self.args.num_attention_heads)
        
        self.multihead_self_attention_sequence = list()
        for i in range(self.args.att_layer):
            self.multihead_self_attention_sequence.append(MultiHeadSelfAttention_Pytorch(self.args.latdim,self.args.num_attention_heads))

        # Initialize FC_Layer instances for meta2 and meta3
        self.meta2_layer = FC_Layer(inDim=self.args.latdim * 3, outDim=self.args.ssldim, useBias=True, activation='leakyRelu', reg=True, reuse=True, name="meta2")
        self.meta3_layer = FC_Layer(inDim=self.args.ssldim, outDim=1, useBias=True, activation='sigmoid', reg=True, reuse=True, name="meta3")
        self.time_FC = FC_Layer(inDim=self.args.latdim, outDim=self.args.latdim, useBias=True, activation='sigmoid', reg=True, reuse=True, name="time_FC")
        for param in self.meta2_layer.parameters():
            self.regParams.append(param)
        for param in self.meta3_layer.parameters():
            self.regParams.append(param)
        for param in self.time_FC.parameters():
            self.regParams.append(param)
    def messagePropagate(self, srclats, mat, type='user'):
        timeEmbed = self.time_FC(self.timeEmbed)
        if not mat.is_coalesced():
            mat = mat.coalesce()
        srcNodes = mat.indices()[1]
        tgtNodes = mat.indices()[0]

        # Ensure srcNodes and tgtNodes are within valid range
        num_nodes = srclats.shape[0]
        srcNodes = torch.clamp(srcNodes, min=0, max=num_nodes - 1)
        tgtNodes = torch.clamp(tgtNodes, min=0, max=num_nodes - 1)

        srcEmbeds = srclats[srcNodes]
        tgtNodes = tgtNodes.unsqueeze(-1).expand_as(srcEmbeds)
        lat = torch.zeros_like(srclats).scatter_add_(0, tgtNodes, srcEmbeds)

        if type == 'user':
            lat = lat[torch.clamp(self.users, min=0, max=lat.shape[0] - 1)]
        else:
            lat = lat[torch.clamp(self.items, min=0, max=lat.shape[0] - 1)]

        return ActivateHelp(lat, self.actFunc)
    
    def edgeDropout(self, mat, keepRate):
        def dropOneMat(mat):
            if not mat.is_coalesced():
                mat = mat.coalesce()
            indices = mat.indices()
            values = mat.values()
            shape = mat.size()
            newVals = F.dropout(values.float(), p=1 - keepRate)
            return torch.sparse_coo_tensor(indices, newVals.type_as(values), shape)
        return dropOneMat(mat)

    def forward(self,
            uids, 
            iids, 
            sequence, 
            mask, 
            uLocs_seq,
            suids,
            siids,
            keepRate=1.0,
            # suLocs_seq  
        ):
        adj = self.handler.trnMat
        idx, data, shape = transToLsts(adj, norm=True)
        self.adj = torch.sparse_coo_tensor(indices=torch.tensor(idx), values=torch.tensor(data), size=shape)
        self.subAdj=list()
        self.subTpAdj=list()
        # #  self.subAdjNp=list()
        for i in range(self.args.graphNum):
            seqadj = self.handler.subMat[i]
            idx, data, shape = transToLsts(seqadj, norm=True)
            # print("1",shape)
            self.subAdj.append(torch.sparse_coo_tensor(indices=torch.tensor(idx), values=torch.tensor(data), size=shape))
            idx, data, shape = transToLsts(transpose(seqadj), norm=True)
            self.subTpAdj.append(torch.sparse_coo_tensor(indices=torch.tensor(idx), values=torch.tensor(data), size=shape))
            # print("2",shape)
        
        user_vector,item_vector=list(),list()
        # user_vector_short,item_vector_short=list(),list()
        # embedding

        for k in range(self.args.graphNum):
            embs0=[self.uEmbed[k]]
            embs1=[self.iEmbed[k]]
            for i in range(self.args.gnn_layer):
                a_emb0= self.messagePropagate(embs1[-1],self.edgeDropout(self.subAdj[k], keepRate),'user')
                a_emb1= self.messagePropagate(embs0[-1],self.edgeDropout(self.subTpAdj[k], keepRate),'item')

                a_emb0 = a_emb0[:embs0[-1].shape[0], :]
                a_emb1 = a_emb1[:embs1[-1].shape[0], :]

                embs0.append(a_emb0+embs0[-1]) 
                embs1.append(a_emb1+embs1[-1]) 
            aligned_embs0 = []
            reference_shape = embs0[0].shape

            for emb in embs0:
                current_shape = emb.shape
                shape_mismatch = any(c != r for c, r in zip(current_shape, reference_shape))
                emb = emb[:reference_shape[0], : ] if shape_mismatch else emb
                aligned_embs0.append(emb)

            # Perform the addition with aligned tensors
            user = torch.stack(aligned_embs0, dim=0).sum(dim=0)
            item = torch.stack(embs1, dim=0).sum(dim=0)  # + torch.tile(timeIEmbed[k], (self.args.item, 1))
            user_vector.append(user)
            item_vector.append(item)
        # now user_vector is [g,u,latdim]
        user_vector=torch.stack(user_vector,dim=0)
        item_vector=torch.stack(item_vector,dim=0)
        user_vector_tensor = user_vector.permute(1, 0, 2)
        item_vector_tensor = item_vector.permute(1, 0, 2)

        # 1. 处理 User Vector
        # rnn 返回 (output, (h_n, c_n))，我们只需要 output
        # output 形状: [batch_size, seq_len, hidden_size]
        user_vector_rnn, _ = self.rnn(user_vector_tensor)
        # 应用 Dropout (PyTorch 会自动处理训练/测试模式)
        user_vector_rnn = self.rnn_dropout(user_vector_rnn)

        # 2. 处理 Item Vector
        # 使用同一个 self.rnn 对象，实现了权值共享
        item_vector_rnn, _ = self.rnn(item_vector_tensor)

        # 应用 Dropout
        item_vector_rnn = self.rnn_dropout(item_vector_rnn)

        # 3. 更新变量
        user_vector_tensor = user_vector_rnn
        item_vector_tensor = item_vector_rnn
        # self.additive_attention0 = self.AdditiveAttention_0(F.layer_norm(user_vector_tensor, user_vector_tensor.shape[-1:]))
        # self.additive_attention1 = self.AdditiveAttention_1(F.layer_norm(item_vector_tensor, item_vector_tensor.shape[-1:]))

        multihead_user_vector = self.MultiHeadSelfAttention_0(
            F.layer_norm(user_vector_tensor, user_vector_tensor.shape[-1:])
        )
        multihead_item_vector = self.MultiHeadSelfAttention_1(
            F.layer_norm(item_vector_tensor, item_vector_tensor.shape[-1:])
        )
        final_user_vector = torch.mean(multihead_user_vector, dim=1)  # + user_vector_long
        final_item_vector = torch.mean(multihead_item_vector, dim=1)  # + item_vector_long
        iEmbed_att=final_item_vector
        # sequence att
        sequence_batch = F.layer_norm(
            torch.matmul(torch.unsqueeze(mask, dim=1), iEmbed_att[sequence]),
            normalized_shape=iEmbed_att[sequence].shape[-1:]
        )
        sequence_batch += F.layer_norm(
            torch.matmul(torch.unsqueeze(mask, dim=1), self.posEmbed[self.pos]),
            normalized_shape=self.posEmbed[self.pos].shape[-1:]
        )
        att_layer=sequence_batch
        for i in range(self.args.att_layer):
            att_layer1=self.multihead_self_attention_sequence[i](F.layer_norm(att_layer, att_layer.shape[-1:]))
            att_layer=ActivateHelp(att_layer1,"leakyRelu")+att_layer
        att_user = torch.sum(att_layer, dim=1)
        # att_user=self.additive_attention0.attention(att_layer)# tf.reduce_sum(att_layer,axis=1)
        pckIlat_att = iEmbed_att[iids]
        pckUlat = final_user_vector[uids]
        pckIlat = final_item_vector[iids]
        preds = torch.sum(pckUlat * pckIlat, dim=-1)
        preds += torch.sum(ActivateHelp(att_user[uLocs_seq],"leakyRelu")* pckIlat_att,dim=-1)
        self.preds_one=list()
        self.final_one=list()
        sslloss = 0    
        user_weight = []
        for i in range(self.args.graphNum):
            meta1 = torch.cat([final_user_vector * user_vector[i], final_user_vector, user_vector[i]], dim=-1)
            meta2 = self.meta2_layer(meta1)
            user_weight.append(torch.squeeze(self.meta3_layer(meta2)))
        user_weight = torch.stack(user_weight, dim=0)    
        for i in range(self.args.graphNum):
            sampNum = suids[i].shape[0] // 2
            pckUlat = final_user_vector[suids[i]]
            pckIlat = final_item_vector[siids[i]]
            pckUweight =  user_weight[i][suids[i]]
            pckIlat_att = iEmbed_att[siids[i]]
            S_final = torch.sum(ActivateHelp(pckUlat* pckIlat, self.actFunc),dim=-1)
            posPred_final = torch.detach(S_final[:sampNum])#.detach()
            negPred_final = torch.detach(S_final[sampNum:])#.detach()
            posweight_final = pckUweight[:sampNum]
            negweight_final = pckUweight[sampNum:]
            S_final = posweight_final*posPred_final-negweight_final*negPred_final
            pckUlat = user_vector[i][suids[i]]
            pckIlat = item_vector[i][siids[i]]
            preds_one = torch.sum(ActivateHelp(pckUlat* pckIlat , self.actFunc), dim=-1)
            posPred = preds_one[:sampNum]
            negPred = preds_one[sampNum:]
            sslloss += torch.sum(torch.maximum(torch.tensor(0.0), 1.0 -S_final * (posPred-negPred)))
            self.preds_one.append(preds_one)
        
        return preds, sslloss
