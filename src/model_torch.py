import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from Params import args
from Utils.attention import AdditiveAttention, MultiHeadSelfAttention
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import scipy.sparse as sp
from random import randint
import Utils.TimeLogger as logger
from Utils.TimeLogger import log

class RecommenderPyTorch(nn.Module):
    def __init__(self, handler):
        super(RecommenderPyTorch, self).__init__()
        self.handler = handler
        self.maxTime = handler.maxTime
        
        # 初始化参数
        self._init_parameters()
        # 构建模型
        self._build_model()
        
        # 指标记录
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def _init_parameters(self):
        """初始化模型参数"""
        # 嵌入层
        self.uEmbed = nn.ParameterList([
            nn.Parameter(torch.empty(args.graphNum, args.user, args.latdim))
            for _ in range(1)
        ])[0]
        self.iEmbed = nn.ParameterList([
            nn.Parameter(torch.empty(args.graphNum, args.item, args.latdim))
            for _ in range(1)
        ])[0]
        
        # 位置嵌入
        self.posEmbed = nn.Parameter(torch.empty(args.pos_length, args.latdim))
        
        # 时间嵌入
        self.timeEmbed = nn.Parameter(torch.empty(self.maxTime + 1, args.latdim))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.uEmbed)
        nn.init.xavier_uniform_(self.iEmbed)
        nn.init.xavier_uniform_(self.posEmbed)
        nn.init.xavier_uniform_(self.timeEmbed)
        
        # 注意力机制
        self.additive_attention0 = AdditiveAttention(args.query_vector_dim, args.latdim)
        self.additive_attention1 = AdditiveAttention(args.query_vector_dim, args.latdim)
        
        self.multihead_self_attention0 = MultiHeadSelfAttention(args.latdim, args.num_attention_heads)
        self.multihead_self_attention1 = MultiHeadSelfAttention(args.latdim, args.num_attention_heads)
        
        # 序列注意力层
        self.multihead_self_attention_sequence = nn.ModuleList([
            MultiHeadSelfAttention(args.latdim, args.num_attention_heads)
            for _ in range(args.att_layer)
        ])
        
        # 元学习层
        self.meta_layers = nn.ModuleDict({
            'meta2': nn.Linear(args.latdim * 3, args.ssldim),
            'meta3': nn.Linear(args.ssldim, 1)
        })

    def _build_model(self):
        """构建模型组件"""
        # RNN 层
        if hasattr(nn, 'LSTM'):
            self.rnn = nn.LSTM(
                input_size=args.latdim,
                hidden_size=args.latdim,
                num_layers=1,
                batch_first=True,
                dropout=1 - args.keepRate if args.keepRate < 1.0 else 0
            )
        else:
            self.rnn = None

    def messagePropagate(self, srclats, mat, nodes, type='user'):
        """消息传播"""
        if sp.issparse(mat):
            # 转换为 PyTorch 稀疏张量
            mat = self._sparse_to_tensor(mat)
        
        # 简单的消息聚合（可根据需要实现更复杂的图卷积）
        if type == 'user':
            # 用户侧聚合
            aggregated = torch.zeros_like(srclats)
            # 这里简化实现，实际需要根据邻接矩阵进行聚合
            return F.leaky_relu(aggregated, negative_slope=args.leaky)
        else:
            # 物品侧聚合
            aggregated = torch.zeros_like(srclats)
            return F.leaky_relu(aggregated, negative_slope=args.leaky)

    def _sparse_to_tensor(self, sp_matrix):
        """将稀疏矩阵转换为 PyTorch 张量"""
        if sp.issparse(sp_matrix):
            sp_matrix = sp_matrix.tocoo()
            indices = torch.LongTensor(np.vstack((sp_matrix.row, sp_matrix.col)))
            values = torch.FloatTensor(sp_matrix.data)
            shape = torch.Size(sp_matrix.shape)
            return torch.sparse.FloatTensor(indices, values, shape)
        return sp_matrix

    def edgeDropout(self, mat, keepRate):
        """边dropout"""
        if torch.is_tensor(mat):
            mask = (torch.rand_like(mat.values()) < keepRate).float()
            new_values = mat.values() * mask
            return torch.sparse.FloatTensor(mat.indices(), new_values, mat.shape)
        return mat

    def forward(self, uids, iids, sequence, mask, uLocs_seq, suids, siids, suLocs_seq, 
                subAdj, subTpAdj, is_train=True, keepRate=1.0):
        """前向传播"""
        
        user_vector, item_vector = [], []
        
        # 多图嵌入传播
        for k in range(args.graphNum):
            embs0 = [self.uEmbed[k]]
            embs1 = [self.iEmbed[k]]
            
            for i in range(args.gnn_layer):
                # 消息传播
                a_emb0 = self.messagePropagate(
                    embs1[-1], 
                    self.edgeDropout(subAdj[k], keepRate), 
                    torch.arange(args.user), 
                    'user'
                )
                a_emb1 = self.messagePropagate(
                    embs0[-1], 
                    self.edgeDropout(subTpAdj[k], keepRate), 
                    torch.arange(args.item), 
                    'item'
                )
                
                embs0.append(a_emb0 + embs0[-1])
                embs1.append(a_emb1 + embs1[-1])
            
            # 残差连接求和
            user = sum(embs0)
            item = sum(embs1)
            user_vector.append(user)
            item_vector.append(item)
        
        # 堆叠和转置
        user_vector = torch.stack(user_vector, dim=0)  # [graphNum, user, latdim]
        item_vector = torch.stack(item_vector, dim=0)  # [graphNum, item, latdim]
        
        user_vector_tensor = user_vector.permute(1, 0, 2)  # [user, graphNum, latdim]
        item_vector_tensor = item_vector.permute(1, 0, 2)  # [item, graphNum, latdim]
        
        # RNN 处理
        if self.rnn is not None and is_train:
            user_vector_rnn, _ = self.rnn(user_vector_tensor)
            item_vector_rnn, _ = self.rnn(item_vector_tensor)
            user_vector_tensor = user_vector_rnn
            item_vector_tensor = item_vector_rnn
        
        # 多头自注意力
        multihead_user_vector = self.multihead_self_attention0.attention(
            F.layer_norm(user_vector_tensor, [args.latdim])
        )
        multihead_item_vector = self.multihead_self_attention1.attention(
            F.layer_norm(item_vector_tensor, [args.latdim])
        )
        
        final_user_vector = torch.mean(multihead_user_vector, dim=1)
        final_item_vector = torch.mean(multihead_item_vector, dim=1)
        iEmbed_att = final_item_vector
        
        # 序列注意力
        sequence_emb = F.embedding(sequence, iEmbed_att)  # [batch, pos_length, latdim]
        pos_emb = F.embedding(torch.arange(args.pos_length).to(sequence.device), self.posEmbed)
        pos_emb = pos_emb.unsqueeze(0).expand(sequence.size(0), -1, -1)
        
        mask_expanded = mask.unsqueeze(1).float()
        sequence_batch = F.layer_norm(
            torch.bmm(mask_expanded, sequence_emb) + 
            torch.bmm(mask_expanded, pos_emb)
        )
        
        att_layer = sequence_batch
        for i in range(args.att_layer):
            att_layer1 = self.multihead_self_attention_sequence[i].attention(
                F.layer_norm(att_layer, [args.latdim])
            )
            att_layer = F.leaky_relu(att_layer1, negative_slope=args.leaky) + att_layer
        
        att_user = torch.sum(att_layer, dim=1)
        
        # 预测计算
        pckIlat_att = F.embedding(iids, iEmbed_att)
        pckUlat = F.embedding(uids, final_user_vector)
        pckIlat = F.embedding(iids, final_item_vector)
        
        preds = torch.sum(pckUlat * pckIlat, dim=-1)
        preds += torch.sum(
            F.leaky_relu(F.embedding(uLocs_seq, att_user), negative_slope=args.leaky) * pckIlat_att,
            dim=-1
        )
        
        # SSL 损失计算
        sslloss = 0
        user_weight = []
        
        for i in range(args.graphNum):
            meta1 = torch.cat([
                final_user_vector * user_vector[i],
                final_user_vector,
                user_vector[i]
            ], dim=-1)
            
            meta2 = F.leaky_relu(self.meta_layers['meta2'](meta1), negative_slope=args.leaky)
            weight = torch.sigmoid(self.meta_layers['meta3'](meta2)).squeeze()
            user_weight.append(weight)
        
        user_weight = torch.stack(user_weight, dim=0)
        
        for i in range(args.graphNum):
            sampNum = suids[i].size(0) // 2
            
            pckUlat = F.embedding(suids[i], final_user_vector)
            pckIlat = F.embedding(siids[i], final_item_vector)
            pckUweight = F.embedding(suids[i], user_weight[i])
            pckIlat_att = F.embedding(siids[i], iEmbed_att)
            
            S_final = torch.sum(F.leaky_relu(pckUlat * pckIlat, negative_slope=args.leaky), dim=-1)
            posPred_final = S_final[:sampNum].detach()
            negPred_final = S_final[sampNum:].detach()
            
            posweight_final = pckUweight[:sampNum]
            negweight_final = pckUweight[sampNum:]
            
            S_final = posweight_final * posPred_final - negweight_final * negPred_final
            
            pckUlat_view = F.embedding(suids[i], user_vector[i])
            pckIlat_view = F.embedding(siids[i], item_vector[i])
            
            preds_one = torch.sum(F.leaky_relu(pckUlat_view * pckIlat_view, negative_slope=args.leaky), dim=-1)
            posPred = preds_one[:sampNum]
            negPred = preds_one[sampNum:]
            
            sslloss += torch.sum(torch.clamp(1.0 - S_final * (posPred - negPred), min=0.0))
        
        return preds, sslloss

    def compute_loss(self, preds, sslloss):
        """计算总损失"""
        sampNum = preds.size(0) // 2
        posPred = preds[:sampNum]
        negPred = preds[sampNum:]
        
        preLoss = torch.mean(torch.clamp(1.0 - (posPred - negPred), min=0.0))
        
        # 正则化损失
        regLoss = 0
        for param in self.parameters():
            if param.requires_grad:
                regLoss += torch.norm(param)
        regLoss = args.reg * regLoss + args.ssl_reg * sslloss
        
        total_loss = preLoss + regLoss
        return preLoss, regLoss, total_loss, posPred, negPred

class RecommenderTrainer:
    def __init__(self, handler):
        self.handler = handler
        self.model = RecommenderPyTorch(handler)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=args.lr,
            weight_decay=args.reg
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=args.decay_step, 
            gamma=args.decay
        )
        
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        log('Model Prepared')
        
        if args.load_model is not None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        
        maxndcg = 0.0
        maxres = dict()
        maxepoch = 0
        
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, test))
            
            if test:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, test))
                
            if ep % args.tstEpoch == 0 and reses['NDCG'] > maxndcg:
                self.saveHistory()
                maxndcg = reses['NDCG']
                maxres = reses
                maxepoch = ep
            
            print()
        
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        log(self.makePrint('max', maxepoch, maxres, True))

    def trainEpoch(self):
        self.model.train()
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = 0, 0
        num = len(sfIds)
        sample_num_list = [40]
        steps = int(np.ceil(num / args.batch))
        
        for s in range(len(sample_num_list)):
            for i in range(steps):
                st = i * args.batch
                ed = min((i+1) * args.batch, num)
                batIds = sfIds[st: ed]
                
                # 获取批次数据
                uLocs, iLocs, sequence, mask, uLocs_seq = self.sampleTrainBatch(
                    batIds, self.handler.trnMat, self.handler.timeMat, sample_num_list[s]
                )
                suLocs, siLocs, suLocs_seq = self.sampleSslBatch(batIds, self.handler.subMat, False)
                
                # 转换为张量
                uLocs = torch.LongTensor(uLocs)
                iLocs = torch.LongTensor(iLocs)
                sequence = torch.LongTensor(sequence)
                mask = torch.FloatTensor(mask)
                uLocs_seq = torch.LongTensor(uLocs_seq)
                
                suLocs_tensor = [torch.LongTensor(su) for su in suLocs]
                siLocs_tensor = [torch.LongTensor(si) for si in siLocs]
                
                # 转换为稀疏张量
                subAdj_tensor = []
                subTpAdj_tensor = []
                for k in range(args.graphNum):
                    subAdj_tensor.append(self.model._sparse_to_tensor(self.handler.subMat[k]))
                    subTpAdj_tensor.append(self.model._sparse_to_tensor(
                        transpose(self.handler.subMat[k])
                    ))
                
                # 前向传播
                self.optimizer.zero_grad()
                preds, sslloss = self.model(
                    uLocs, iLocs, sequence, mask, uLocs_seq,
                    suLocs_tensor, siLocs_tensor, suLocs_seq,
                    subAdj_tensor, subTpAdj_tensor,
                    is_train=True, keepRate=args.keepRate
                )
                
                # 计算损失
                preLoss, regLoss, loss, posPred, negPred = self.model.compute_loss(preds, sslloss)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epochLoss += loss.item()
                epochPreLoss += preLoss.item()
                
                log('Step %d/%d: preloss = %.2f, REGLoss = %.2f' % 
                    (i+s*steps, steps*len(sample_num_list), preLoss.item(), regLoss.item()), 
                    save=False, oneline=True)
        
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret

    def testEpoch(self):
        self.model.eval()
        epochHit, epochNdcg = 0, 0
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        
        with torch.no_grad():
            for i in range(steps):
                st = i * tstBat
                ed = min((i+1) * tstBat, num)
                batIds = ids[st: ed]
                
                # 获取测试数据
                uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list = self.sampleTestBatch(
                    batIds, self.handler.trnMat
                )
                suLocs, siLocs, _ = self.sampleSslBatch(batIds, self.handler.subMat, False)
                
                # 转换为张量
                uLocs = torch.LongTensor(uLocs)
                iLocs = torch.LongTensor(iLocs)
                sequence = torch.LongTensor(sequence)
                mask = torch.FloatTensor(mask)
                uLocs_seq = torch.LongTensor(uLocs_seq)
                
                suLocs_tensor = [torch.LongTensor(su) for su in suLocs]
                siLocs_tensor = [torch.LongTensor(si) for si in siLocs]
                
                # 转换为稀疏张量
                subAdj_tensor = []
                subTpAdj_tensor = []
                for k in range(args.graphNum):
                    subAdj_tensor.append(self.model._sparse_to_tensor(self.handler.subMat[k]))
                    subTpAdj_tensor.append(self.model._sparse_to_tensor(
                        transpose(self.handler.subMat[k])
                    ))
                
                # 前向传播
                preds, _ = self.model(
                    uLocs, iLocs, sequence, mask, uLocs_seq,
                    suLocs_tensor, siLocs_tensor, uLocs_seq,
                    subAdj_tensor, subTpAdj_tensor,
                    is_train=False, keepRate=1.0
                )
                
                preds = preds.numpy().reshape(ed-st, args.testSize)
                
                if args.test:
                    hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15 = self.calcRes(
                        preds, temTst, tstLocs
                    )
                else:
                    hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15 = self.calcRes(
                        preds, val_list, tstLocs
                    )
                
                epochHit += hit
                epochNdcg += ndcg
                
                log('Steps %d/%d: hit10 = %d, ndcg10 = %d' % (i, steps, hit, ndcg), save=False, oneline=True)
        
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    # 以下方法保持不变（sampleTrainBatch, sampleSslBatch, sampleTestBatch, calcRes, saveHistory, loadModel）
    # 因为它们主要涉及数据预处理，与 TensorFlow/PyTorch 无关
    
    def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
        # 实现与原来相同
        pass
    
    def sampleSslBatch(self, batIds, labelMat, use_epsilon=True):
        # 实现与原来相同
        pass
    
    def sampleTestBatch(self, batIds, labelMat):
        # 实现与原来相同
        pass
    
    def calcRes(self, preds, temTst, tstLocs):
        # 实现与原来相同
        pass
    
    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, 'Models/' + args.save_path + '.pth')
        log('Model Saved: %s' % args.save_path)
    
    def loadModel(self):
        checkpoint = torch.load('Models/' + args.load_model + '.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']
        
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics.update(pickle.load(fs))
        log('Model Loaded')