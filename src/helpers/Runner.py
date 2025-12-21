import torch
# import torch.optim as optim
import numpy as np
from helpers.BaseRunner import BaseRunner
import pickle
import os
from Utils_.utils import *
from random import randint

class Runner(BaseRunner):
    def __init__(self, args):
        super(Runner, self).__init__(args)
        # self.model = model
        # self.handler = model.handler
        # self.args = model.args
        # self.regParams = model.regParams
        self.args = args
        
        print('USER', self.args.user, 'ITEM', self.args.item)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
    
    def print_res(self, phase):
        self.model = phase.model
        self.handler = phase.handler
        self.args = phase.args
        phase = phase.phase
        if phase == 'dev':
            reses = self.testEpoch(False)
        elif phase == 'test':
            reses = self.testEpoch(True)
        return self.makePrint(phase, 0, reses, True)
    
    def train(self, data_dict):
        self.model = data_dict['train'].model
        self.handler = data_dict['train'].handler
        self.args = data_dict['train'].args
        globalStep = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.decay_step, gamma=self.args.decay)
        return self.run(globalStep, optimizer, scheduler)
    
    def run(self, globalStep, optimizer, scheduler):
        print('Model Prepared')
        if self.args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * self.args.tstEpoch - (self.args.tstEpoch - 1)
        else:
            stloc = 0
            print('Variables Inited')
        maxndcg=0.0
        maxres=dict()
        maxepoch=0
        
        for ep in range(stloc, self.args.epoch):
            test = (ep % self.args.tstEpoch == 0)
            reses = self.trainEpoch(globalStep, optimizer, scheduler)
            print(self.makePrint('Train', ep, reses, test))
            if test:
                reses = self.testEpoch(True)
                print(self.makePrint('Test', ep, reses, test))
                # self.makePrint('Test', ep, reses, test)
            if ep % self.args.tstEpoch == 0 and reses['NDCG']>maxndcg:
                self.saveHistory()
                maxndcg=reses['NDCG']
                maxres=reses
                maxepoch=ep
            print()
        reses = self.testEpoch()
        print(self.makePrint('Test', self.args.epoch, reses, True))
        print(self.makePrint('max', maxepoch, maxres, True))
        # 保存最终模型
        self.saveHistory(final=True)
    
    def trainEpoch(self, globalStep, optimizer, scheduler):
        num = self.args.user
        sfIds = np.random.permutation(num)[:self.args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        sample_num_list=[40]        
        steps = int(np.ceil(num / self.args.batch))
        for s in range(len(sample_num_list)):
            for i in range(steps):
                st = i * self.args.batch
                ed = min((i+1) * self.args.batch, num)
                batIds = sfIds[st: ed]

                uLocs, iLocs, sequence, mask, uLocs_seq= self.sampleTrainBatch(batIds, self.handler.trnMat, self.handler.timeMat, sample_num_list[s])
                # esuLocs, esiLocs, epsilon = self.sampleSslBatch(batIds, self.handler.subadj)
                suLocs, siLocs, suLocs_seq = self.sampleSslBatch(batIds, self.handler.subMat, False)

                # 3. Transfer Data to Device (GPU/CPU)
                # 将采样得到的数据赋值给 self 变量，因为 ours_pytorch 方法中使用了 self.uids 等
                uids = torch.tensor(uLocs, dtype=torch.long)
                iids = torch.tensor(iLocs, dtype=torch.long)
                sequence = torch.tensor(np.array(sequence), dtype=torch.long)
                mask = torch.tensor(np.array(mask), dtype=torch.float32)
                uLocs_seq = torch.tensor(uLocs_seq, dtype=torch.long)

                # 处理 SSL 数据的列表结构
                suids = [torch.tensor(x, dtype=torch.long) for x in suLocs]
                siids = [torch.tensor(x, dtype=torch.long) for x in siLocs]
                suLocs_seq = [torch.tensor(x, dtype=torch.long) for x in suLocs_seq]

                # 4. Optimization Step
                optimizer.zero_grad()

                # 前向传播：调用 ours_pytorch 获取预测值和 SSL 损失
                preds, sslloss = self.model(
                    uids, 
                    iids, 
                    sequence, 
                    mask, 
                    uLocs_seq,
                    suids,
                    siids,
                    keepRate=self.args.keepRate,
                    # self.suLocs_seq    
                )

                # 计算 Main Loss (Pairwise Ranking Loss / BPR-like)
                sampNum = uids.shape[0] // 2
                posPred = preds[:sampNum]
                negPred = preds[sampNum:]
                # preLoss = mean(max(0, 1 - (pos - neg)))
                preLoss = torch.mean(torch.clamp(1.0 - (posPred - negPred), min=0.0))

                # 计算 Regularization Loss
                regLoss = self.args.reg * self.model.get_reg_loss() + self.args.ssl_reg * sslloss
                
                loss = preLoss + regLoss

                # 反向传播
                loss.backward()
                optimizer.step()

                # 5. Logging
                epochLoss += loss.item()
                epochPreLoss += preLoss.item()

                epochLoss += loss
                epochPreLoss += preLoss
                print('Step %d/%d: preloss = %.2f, REGLoss = %.2f         ' % (i+s*steps, steps*len(sample_num_list), preLoss, regLoss))
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret
    
    def testEpoch(self, sign=True):
        epochHit, epochNdcg = [0] * 2
        epochHit5, epochNdcg5 = [0] * 2
        epochHit20, epochNdcg20 = [0] * 2
        epochHit1, epochNdcg1 = [0] * 2
        epochHit15, epochNdcg15 = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = self.args.batch
        steps = int(np.ceil(num / tstBat))
        # np.random.seed(100)
        for i in range(steps):
            st = i * tstBat
            ed = min((i+1) * tstBat, num)
            batIds = ids[st: ed]
            uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list = self.sampleTestBatch(batIds, self.handler.trnMat, sign)
            suLocs, siLocs, _ = self.sampleSslBatch(batIds, self.handler.subMat, False)
            uids = torch.tensor(uLocs, dtype=torch.long)
            iids = torch.tensor(iLocs, dtype=torch.long)
            sequence = torch.tensor(np.array(sequence), dtype=torch.long)
            mask = torch.tensor(np.array(mask), dtype=torch.float32)
            uLocs_seq = torch.tensor(uLocs_seq, dtype=torch.long)

            # 处理 SSL 数据的列表结构
            suids = [torch.tensor(x, dtype=torch.long) for x in suLocs]
            siids = [torch.tensor(x, dtype=torch.long) for x in siLocs]
            # suLocs_seq = [torch.tensor(x, dtype=torch.long) for x in suLocs_seq]
            
            preds, _ = self.model(
                    uids, 
                    iids, 
                    sequence, 
                    mask, 
                    uLocs_seq,
                    suids,
                    siids
                )
            if(self.args.uid!=-1):
                print(preds[self.args.uid].detach().cpu().numpy())
            preds_ = preds.detach().cpu().numpy()
            if(sign):
                hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1,  hit15, ndcg15= self.calcRes(np.reshape(preds_, [ed-st, self.args.testSize]), temTst, tstLocs)
            else:
                hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1,  hit15, ndcg15= self.calcRes(np.reshape(preds_, [ed-st, self.args.testSize]), val_list, tstLocs)
            epochHit += hit
            epochNdcg += ndcg
            epochHit5 += hit5
            epochNdcg5 += ndcg5
            epochHit20 += hit20
            epochNdcg20 += ndcg20
            epochHit15 += hit15
            epochNdcg15 += ndcg15
            epochHit1 += hit1
            epochNdcg1 += ndcg1
            print('Steps %d/%d: hit10 = %d, ndcg10 = %d' % (i, steps, hit, ndcg))
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        print("epochNdcg1:{},epochHit1:{},epochNdcg5:{},epochHit5:{}".format(epochNdcg1/ num,epochHit1/ num,epochNdcg5/ num,epochHit5/ num))
        print("epochNdcg15:{},epochHit15:{},epochNdcg20:{},epochHit20:{}".format(epochNdcg15/ num,epochHit15/ num,epochNdcg20/ num,epochHit20/ num))
        return ret
    
    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, self.args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret
    
    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        hit1 = 0
        ndcg1 = 0
        hit5=0
        ndcg5=0
        hit20=0
        ndcg20=0
        hit15=0
        ndcg15=0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:self.args.shoot]))
            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
            shoot = list(map(lambda x: x[1], predvals[:5]))
            if temTst[j] in shoot:
                hit5 += 1
                ndcg5 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
            shoot = list(map(lambda x: x[1], predvals[:20]))    
            if temTst[j] in shoot:
                hit20 += 1
                ndcg20 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))    
        return hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15
    
    def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
        temTst = self.handler.tstInt[batIds]
        temLabel=labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * train_sample_num
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        uLocs_seq = [None]* temlen
        sequence = [None] * self.args.batch
        mask = [None]*self.args.batch
        cur = 0                
        # utime = [[list(),list()] for x in range(self.args.graphNum)]
        for i in range(batch):
            posset=self.handler.sequence[batIds[i]][:-1]
            # posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(train_sample_num, len(posset))
            choose=1
            if sampNum == 0:
                poslocs = [np.random.choice(self.args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = []
                # choose = 1
                choose = randint(1,max(min(self.args.pred_num+1,len(posset)-3),1))
                poslocs.extend([posset[-choose]]*sampNum)
                neglocs = negSamp(temLabel[i], sampNum, self.args.item, [self.handler.sequence[batIds[i]][-1],temTst[i]], self.handler.item_with_pop)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
                uLocs_seq[cur] = uLocs_seq[cur+temlen//2] = i
                iLocs[cur] = posloc
                iLocs[cur+temlen//2] = negloc
                cur += 1
            sequence[i]=np.zeros(self.args.pos_length,dtype=int)
            mask[i]=np.zeros(self.args.pos_length)
            posset=posset[:-choose]# self.handler.sequence[batIds[i]][:-choose]
            if(len(posset)<=self.args.pos_length):
                sequence[i][-len(posset):]=posset
                mask[i][-len(posset):]=1
            else:
                sequence[i]=posset[-self.args.pos_length:]
                mask[i]+=1
        uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
        uLocs_seq = uLocs_seq[:cur] + uLocs_seq[temlen//2: temlen//2 + cur]
        if(batch<self.args.batch):
            for i in range(batch,self.args.batch):
                sequence[i]=np.zeros(self.args.pos_length,dtype=int)
                mask[i]=np.zeros(self.args.pos_length)
        return uLocs, iLocs, sequence,mask, uLocs_seq# ,utime

    def sampleSslBatch(self, batIds, labelMat, use_epsilon=True):
        temLabel=list()
        for k in range(self.args.graphNum):    
            temLabel.append(labelMat[k][batIds].toarray())
        batch = len(batIds)
        temlen = batch * 2 * self.args.sslNum
        uLocs = [[None] * temlen] * self.args.graphNum
        iLocs = [[None] * temlen] * self.args.graphNum
        uLocs_seq = [[None] * temlen] * self.args.graphNum
        # epsilon=[[None] * temlen] * self.args.graphNum
        for k in range(self.args.graphNum):    
            cur = 0                
            for i in range(batch):
                posset = np.reshape(np.argwhere(temLabel[k][i]!=0), [-1])
                # print(posset)
                sslNum = min(self.args.sslNum, len(posset)//2)# len(posset)//4# 
                if sslNum == 0:
                    poslocs = [np.random.choice(self.args.item)]
                    neglocs = [poslocs[0]]
                else:
                    all = np.random.choice(posset, sslNum*2) #- self.args.user
                    # print(all)
                    poslocs = all[:sslNum]
                    neglocs = all[sslNum:]
                for j in range(sslNum):
                    posloc = poslocs[j]
                    negloc = neglocs[j]            
                    uLocs[k][cur] = uLocs[k][cur+1] = batIds[i]
                    uLocs_seq[k][cur] = uLocs_seq[k][cur+1] = i
                    iLocs[k][cur] = posloc
                    iLocs[k][cur+1] = negloc
                    cur += 2
            uLocs[k]=uLocs[k][:cur]
            iLocs[k]=iLocs[k][:cur]
            uLocs_seq[k]=uLocs_seq[k][:cur]
        return uLocs, iLocs, uLocs_seq

    def sampleTestBatch(self, batIds, labelMat, sign): # labelMat=TrainMat(adj)
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        temlen = batch * self.args.testSize# self.args.item
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        uLocs_seq = [None] * temlen
        tstLocs = [None] * batch
        sequence = [None] * self.args.batch
        mask = [None]*self.args.batch
        cur = 0
        val_list=[None]*self.args.batch
        for i in range(batch):
            if(sign):
                posloc = temTst[i]
            else:
                posloc = self.handler.sequence[batIds[i]][-1]
                val_list[i]=posloc
            rdnNegSet = np.array(self.handler.test_dict[batIds[i]+1][:self.args.testSize-1])-1
            locset = np.concatenate((rdnNegSet, np.array([posloc])))
            tstLocs[i] = locset
            for j in range(len(locset)):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                uLocs_seq[cur] = i
                cur += 1
            sequence[i]=np.zeros(self.args.pos_length,dtype=int)
            mask[i]=np.zeros(self.args.pos_length)
            if(self.args.test==True):
                posset=self.handler.sequence[batIds[i]]
            else:
                posset=self.handler.sequence[batIds[i]][:-1]
            # posset=self.handler.sequence[batIds[i]]
            if(len(posset)<=self.args.pos_length):
                sequence[i][-len(posset):]=posset
                mask[i][-len(posset):]=1
            else:
                sequence[i]=posset[-self.args.pos_length:]
                mask[i]+=1
        if(batch<self.args.batch):
            for i in range(batch,self.args.batch):
                sequence[i]=np.zeros(self.args.pos_length,dtype=int)
                mask[i]=np.zeros(self.args.pos_length)
        return uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list
    
    def saveHistory(self, final=False):
        if self.args.epoch == 0:
            return
        
        # 创建保存目录
        os.makedirs('History', exist_ok=True)
        os.makedirs('Models', exist_ok=True)
        
        # 保存训练历史指标
        with open('History/' + self.args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)
        
        # 保存 PyTorch 模型
        save_path = 'Models/' + self.args.save_path
        if final:
            save_path += '_final'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics,
            'args': self.args
        }, save_path + '.pth')
        
        print('Model Saved: %s' % (save_path + '.pth'))

    def loadModel(self):
        # 加载 PyTorch 模型
        load_path = 'Models/' + self.args.load_model + '.pth'
        
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            
            print('Model Loaded from: %s' % load_path)
        else:
            # 兼容旧格式：尝试加载历史文件
            his_path = 'History/' + self.args.load_model + '.his'
            if os.path.exists(his_path):
                with open(his_path, 'rb') as fs:
                    self.metrics = pickle.load(fs)
            print('Model Loaded (history only): %s' % his_path)
        