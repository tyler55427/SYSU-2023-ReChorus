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
        self.num_users = args.user
        self.num_items = args.item
        self.device = args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.leaky = args.leaky if hasattr(args, 'leaky') else 0.5
        
        self.prepareModel()

    def prepareModel(self):
        """准备模型组件"""
        self.maxTime = self.handler.maxTime
        self.actFunc = 'leakyRelu'
        
        # Embeddings
        self.uEmbed = nn.Parameter(torch.empty(self.args.graphNum, self.num_users, self.args.latdim))
        self.iEmbed = nn.Parameter(torch.empty(self.args.graphNum, self.num_items, self.args.latdim))
        self.posEmbed = nn.Parameter(torch.empty(self.args.pos_length, self.args.latdim))
        self.timeEmbed = nn.Parameter(torch.empty(self.maxTime + 1, self.args.latdim))
        
        # 初始化 - 对于3D张量需要按2D切片初始化，以匹配TF的xavier_initializer行为
        for i in range(self.args.graphNum):
            nn.init.xavier_uniform_(self.uEmbed.data[i])
            nn.init.xavier_uniform_(self.iEmbed.data[i])
        nn.init.xavier_uniform_(self.posEmbed)
        nn.init.xavier_uniform_(self.timeEmbed)
        
        # Time embedding projection
        self.time_fc = nn.Linear(self.args.latdim, self.args.latdim)
        nn.init.xavier_uniform_(self.time_fc.weight)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(self.args.latdim, self.args.latdim, num_layers=1, batch_first=True, dropout=0)
        
        # Multi-head attention layers
        self.multihead_self_attention0 = MultiHeadSelfAttention(self.args.latdim, self.args.num_attention_heads)
        self.multihead_self_attention1 = MultiHeadSelfAttention(self.args.latdim, self.args.num_attention_heads)
        
        # Sequence attention layers
        self.multihead_self_attention_sequence = nn.ModuleList([
            MultiHeadSelfAttention(self.args.latdim, self.args.num_attention_heads)
            for _ in range(self.args.att_layer)
        ])
        
        # Meta learning layers for SSL
        self.meta2_fc = nn.Linear(self.args.latdim * 3, self.args.ssldim)
        self.meta3_fc = nn.Linear(self.args.ssldim, 1)
        nn.init.xavier_uniform_(self.meta2_fc.weight)
        nn.init.xavier_uniform_(self.meta3_fc.weight)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.args.latdim)
        
        # 预处理邻接矩阵
        self._prepare_adjacency_matrices()

    def _prepare_adjacency_matrices(self):
        """预处理邻接矩阵，转换为PyTorch稀疏张量"""
        self.subAdj_list = []
        self.subTpAdj_list = []
        
        for i in range(self.args.graphNum):
            seqadj = self.handler.subMat[i]
            idx, data, shape = transToLsts(seqadj, norm=True)
            # 存储为元组 (indices, values, shape)
            indices = torch.tensor(idx.T, dtype=torch.long)
            values = torch.tensor(data, dtype=torch.float32)
            self.subAdj_list.append((indices, values, shape))
            
            idx, data, shape = transToLsts(transpose(seqadj), norm=True)
            indices = torch.tensor(idx.T, dtype=torch.long)
            values = torch.tensor(data, dtype=torch.float32)
            self.subTpAdj_list.append((indices, values, shape))

    def edge_dropout(self, adj_indices, adj_values, adj_shape, keep_rate):
        """边dropout - 非inplace版本"""
        if self.training and keep_rate < 1.0:
            mask = torch.bernoulli(torch.ones_like(adj_values) * keep_rate)
            new_values = adj_values * mask / (keep_rate + 1e-8)
            return adj_indices, new_values, adj_shape
        return adj_indices, adj_values, adj_shape

    def message_propagate(self, srclats, adj_indices, adj_values, adj_shape, target_type='user'):
        """消息传播 - 内存优化版本"""
        out_size = self.num_users if target_type == 'user' else self.num_items
        device = srclats.device
        
        if adj_indices.shape[1] == 0:
            return torch.zeros(out_size, self.args.latdim, device=device)
        
        # 获取源节点和目标节点索引
        src_nodes = adj_indices[1].clamp(0, srclats.shape[0] - 1)
        tgt_nodes = adj_indices[0].clamp(0, out_size - 1)
        
        # 获取源节点嵌入
        src_embeds = srclats[src_nodes].to(device)
        
        # 使用scatter_add进行聚合
        lat = torch.zeros(out_size, self.args.latdim, device=device)
        tgt_expanded = tgt_nodes.unsqueeze(-1).expand(-1, self.args.latdim).to(device)
        lat = lat.scatter_add(0, tgt_expanded, src_embeds)
        
        # LeakyReLU
        return F.leaky_relu(lat, negative_slope=self.leaky)

    def forward(self,
            uids, 
            iids, 
            sequence, 
            mask, 
            uLocs_seq,
            suids,
            siids,
            keepRate=1.0,
        ):
        """前向传播 - 内存优化版本"""
        batch_size = sequence.shape[0]
        device = sequence.device
        is_train = self.training
        
        # 确保邻接矩阵在正确的设备上
        if not hasattr(self, '_adj_on_device') or self._adj_on_device != device:
            self._move_adj_to_device(device)
            self._adj_on_device = device
        
        # 预分配列表
        user_vector_list = [None] * self.args.graphNum
        item_vector_list = [None] * self.args.graphNum
        
        # 位置编码
        pos = torch.arange(self.args.pos_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # GNN传播
        for k in range(self.args.graphNum):
            embs0 = [self.uEmbed[k]]
            embs1 = [self.iEmbed[k]]
            
            adj_k = self.subAdj_list[k]
            tpadj_k = self.subTpAdj_list[k]
            
            for i in range(self.args.gnn_layer):
                # Edge dropout
                if is_train:
                    adj_indices, adj_values, adj_shape = self.edge_dropout(
                        adj_k[0], adj_k[1], adj_k[2], keepRate)
                    tpadj_indices, tpadj_values, tpadj_shape = self.edge_dropout(
                        tpadj_k[0], tpadj_k[1], tpadj_k[2], keepRate)
                else:
                    adj_indices, adj_values, adj_shape = adj_k
                    tpadj_indices, tpadj_values, tpadj_shape = tpadj_k
                
                a_emb0 = self.message_propagate(embs1[-1], adj_indices, adj_values, adj_shape, 'user')
                a_emb1 = self.message_propagate(embs0[-1], tpadj_indices, tpadj_values, tpadj_shape, 'item')
                
                # 确保形状匹配
                a_emb0 = a_emb0[:embs0[-1].shape[0]]
                a_emb1 = a_emb1[:embs1[-1].shape[0]]
                
                embs0.append(a_emb0 + embs0[-1])
                embs1.append(a_emb1 + embs1[-1])
            
            # 对齐并求和
            user = embs0[0].clone()
            for emb in embs0[1:]:
                user = user + emb
            item = embs1[0].clone()
            for emb in embs1[1:]:
                item = item + emb
            
            user_vector_list[k] = user
            item_vector_list[k] = item
            
            del embs0, embs1
        
        # Stack: [graphNum, num_users, latdim]
        user_vector = torch.stack(user_vector_list, dim=0)
        item_vector = torch.stack(item_vector_list, dim=0)
        
        # Transpose: [num_users, graphNum, latdim]
        user_vector_tensor = user_vector.permute(1, 0, 2)
        item_vector_tensor = item_vector.permute(1, 0, 2)
        
        # LSTM处理
        user_vector_rnn, _ = self.lstm(user_vector_tensor)
        item_vector_rnn, _ = self.lstm(item_vector_tensor)
        
        if is_train and keepRate < 1.0:
            user_vector_rnn = F.dropout(user_vector_rnn, p=1.0-keepRate, training=True)
            item_vector_rnn = F.dropout(item_vector_rnn, p=1.0-keepRate, training=True)
        
        user_vector_tensor = user_vector_rnn
        item_vector_tensor = item_vector_rnn
        
        # Multi-head attention
        multihead_user_vector = self.multihead_self_attention0(
            self.layer_norm(user_vector_tensor))
        multihead_item_vector = self.multihead_self_attention1(
            self.layer_norm(item_vector_tensor))
        
        # Mean pooling
        final_user_vector = multihead_user_vector.mean(dim=1)
        final_item_vector = multihead_item_vector.mean(dim=1)
        iEmbed_att = final_item_vector

        device = iEmbed_att.device
        sequence = sequence.to(device)
        mask = mask.to(device)
        pos = pos.to(device)
        
        # Sequence attention
        seq_item_embed = iEmbed_att[sequence]  # [batch, pos_length, latdim]
        pos_embed = self.posEmbed[pos]  # [batch, pos_length, latdim]
        
        mask_for_matmul = mask.unsqueeze(1)  # [batch, 1, pos_length]
        sequence_batch = self.layer_norm(torch.bmm(mask_for_matmul, seq_item_embed))
        sequence_batch = sequence_batch + self.layer_norm(torch.bmm(mask_for_matmul, pos_embed))
        
        att_layer = sequence_batch
        for i in range(self.args.att_layer):
            att_layer1 = self.multihead_self_attention_sequence[i](self.layer_norm(att_layer))
            att_layer = F.leaky_relu(att_layer1, negative_slope=self.leaky) + att_layer
        
        att_user = att_layer.sum(dim=1)  # [batch, latdim]
        
        # 预测
        pckIlat_att = iEmbed_att[iids]
        pckUlat = final_user_vector[uids]
        pckIlat = final_item_vector[iids]
        
        preds = (pckUlat * pckIlat).sum(dim=-1)
        preds = preds + (F.leaky_relu(att_user[uLocs_seq], negative_slope=self.leaky) * pckIlat_att).sum(dim=-1)
        
        # SSL Loss计算
        sslloss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 预计算user_weight
        user_weight_list = []
        for i in range(self.args.graphNum):
            with torch.no_grad() if not is_train else torch.enable_grad():
                meta1 = torch.cat([
                    final_user_vector * user_vector[i],
                    final_user_vector,
                    user_vector[i]
                ], dim=-1)
            meta2 = F.leaky_relu(self.meta2_fc(meta1), negative_slope=self.leaky)
            weight = torch.sigmoid(self.meta3_fc(meta2)).squeeze(-1)
            user_weight_list.append(weight)
            del meta1, meta2
        
        user_weight = torch.stack(user_weight_list, dim=0)
        del user_weight_list
        
        for i in range(self.args.graphNum):
            if len(suids[i]) == 0:
                continue
            suids_i = suids[i]
            siids_i = siids[i]
            
            sampNum = len(suids_i) // 2
            if sampNum == 0:
                continue
            
            pckUlat_ssl = final_user_vector[suids_i]
            pckIlat_ssl = final_item_vector[siids_i]
            pckUweight = user_weight[i][suids_i]
            
            S_final = F.leaky_relu(pckUlat_ssl * pckIlat_ssl, negative_slope=self.leaky).sum(dim=-1)
            posPred_final = S_final[:sampNum].detach()
            negPred_final = S_final[sampNum:].detach()
            posweight_final = pckUweight[:sampNum]
            negweight_final = pckUweight[sampNum:]
            S_final_weighted = posweight_final * posPred_final - negweight_final * negPred_final
            
            pckUlat_graph = user_vector[i][suids_i]
            pckIlat_graph = item_vector[i][siids_i]
            preds_one = F.leaky_relu(pckUlat_graph * pckIlat_graph, negative_slope=self.leaky).sum(dim=-1)
            posPred = preds_one[:sampNum]
            negPred = preds_one[sampNum:]
            
            sslloss = sslloss + torch.clamp(1.0 - S_final_weighted * (posPred - negPred), min=0.0).sum()
        
        return preds, sslloss

    def _move_adj_to_device(self, device):
        """将邻接矩阵移动到指定设备"""
        for i in range(len(self.subAdj_list)):
            indices, values, shape = self.subAdj_list[i]
            self.subAdj_list[i] = (indices.to(device), values.to(device), shape)
            
            indices, values, shape = self.subTpAdj_list[i]
            self.subTpAdj_list[i] = (indices.to(device), values.to(device), shape)

    def get_reg_loss(self):
        """获取正则化损失"""
        device = self.uEmbed.device
        reg_loss = torch.tensor(0.0, device=device)
        # Embedding参数正则化
        reg_loss = reg_loss + torch.sum(self.uEmbed ** 2)
        reg_loss = reg_loss + torch.sum(self.iEmbed ** 2)
        reg_loss = reg_loss + torch.sum(self.posEmbed ** 2)
        reg_loss = reg_loss + torch.sum(self.timeEmbed ** 2)
        # FC层权重正则化
        reg_loss = reg_loss + torch.sum(self.meta2_fc.weight ** 2)
        reg_loss = reg_loss + torch.sum(self.meta3_fc.weight ** 2)
        return reg_loss

    class Dataset(BaseModel.Dataset):
        def __init__(self, model, handler, phase):
            self.args = model.args
            self.model = model
            self.handler = handler
            self.phase = phase
            self.data = {}
            
        def prepare(self):
            pass
            
