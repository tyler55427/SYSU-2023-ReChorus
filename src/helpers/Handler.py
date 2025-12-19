import pickle
import numpy as np
from numpy.ma.core import negative
# from scipy.sparse import csr_matrix
import scipy.sparse as sp
import os
# import csv
# import random
from helpers.BaseReader import BaseReader
# import torch
import csv
from scipy.sparse import csr_matrix
from Utils_.utils import *

class DataHandler(BaseReader):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
        parser.add_argument('--percent', default=0.0, type=float, help='percent of noise for noise robust test')
        parser.add_argument('--slot', default=1, type=float, help='length of time slots')
        parser.add_argument('--graphSampleN', default=15000, type=int, help='use 25000 for training and 200000 for testing, empirically')
        parser.add_argument('--graphNum', default=8, type=int, help='number of graphs based on time series')
        parser.add_argument('--leaky', default=0.5, type=float, help='slope for leaky relu')
        parser.add_argument('--batch', default=512, type=int, help='batch size')
        parser.add_argument('--testbatch', default=64, type=int, help='test batch size')
        parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
        parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
        parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
        parser.add_argument('--latdim', default=64, type=int, help='embedding size')
        parser.add_argument('--ssldim', default=32, type=int, help='user weight embedding size')
        parser.add_argument('--rank', default=4, type=int, help='embedding size')
        parser.add_argument('--memosize', default=2, type=int, help='memory size')
        parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
        parser.add_argument('--testSize', default=100, type=int, help='size for test')
        parser.add_argument('--sslNum', default=20, type=int, help='batch size for ssl')
        parser.add_argument('--query_vector_dim', type=int, default=64, help='number of query vector\'s dimension [default: 64]')
        parser.add_argument('--num_attention_heads', type=int, default=16, help='number of num attention heads [default: 16]')
        parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')
        parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
        parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
        parser.add_argument('--load_model', default=None, help='model name to load')
        parser.add_argument('--shoot', default=10, type=int, help='K of top k')
        parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
        parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
        parser.add_argument('--mult', default=100, type=float, help='multiplier for the result')
        parser.add_argument('--keepRate', default=0.5, type=float, help='rate for dropout')
        parser.add_argument('--divSize', default=10000, type=int, help='div size for smallTestEpoch')
        parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
        parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
        parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')
        parser.add_argument('--hyperReg', default=1e-4, type=float, help='regularizer for hyper connections')
        parser.add_argument('--temp', default=1, type=float, help='temperature in ssl loss')
        parser.add_argument('--ssl_reg', default=1e-4, type=float, help='reg weight for ssl loss')
        parser.add_argument('--pos_length', default=200, type=int, help='max length of a sequence')
        parser.add_argument('--att_size', default=12000, type=int, help='max size of multi att')
        parser.add_argument('--att_layer', default=4, type=int, help='layer number of multi att')
        parser.add_argument('--pred_num', default=5, type=int, help='pred number of train')
        parser.add_argument('--nfs', default=False, type=bool, help='load from nfs')
        parser.add_argument('--test', default=True, type=bool, help='test or val')
        parser.add_argument('--ssl', default=True, type=bool, help='use self-supervised learning')
        parser.add_argument('--uid', default=0, type=int, help='show user score')
        return parser

    def __init__(self, args):
        self.args = args
        self.args.decay_step = args.trnNum//args.batch
        if self.args.data == 'yelp':
            predir = './data/Yelp/'
        elif self.args.data == 'gowalla':
            predir = './data/gowalla/'
        elif self.args.data == 'amazon':
            predir = './data/amazon/'
        else:
            predir='./data/'+self.args.data+'/'
        self.predir = predir
        self.trnfile = predir + 'trn_mat_time'
        self.tstfile = predir + 'tst_int'
        self.sequencefile=predir+'sequence'
        self.test_dictfile=predir+'test_dict'
        self.sign = (os.path.exists(self.trnfile) == False)
        self.LoadData()
    def LoadData(self):
        if self.sign==False:
            if self.args.percent > 1e-8:
                print('noised')
                with open(self.predir + 'noise_%.2f' % self.args.percent, 'rb') as fs:
                    trnMat = pickle.load(fs)
            else:
                with open(self.trnfile, 'rb') as fs:
                    # print(pickle.load(fs))
                    trnMat = pickle.load(fs)# (pickle.load(fs) != 0).astype(np.float32)
            # test set
            with open(self.tstfile, 'rb') as fs:
                tstInt = np.array(pickle.load(fs))
            with open(self.sequencefile, 'rb') as fs:
                self.sequence = pickle.load(fs)
            if os.path.isfile(self.test_dictfile):
                with open(self.test_dictfile, 'rb') as fs:
                    self.test_dict = pickle.load(fs)
            print("tstInt",tstInt)
            tstStat = (tstInt != None)
            print("tstStat",tstStat,len(tstStat))
            tstUsrs = np.reshape(np.argwhere(tstStat != False), [-1])
            print("tstUsrs",tstUsrs,len(tstUsrs))
            self.trnMat = trnMat[0]
            def generate_rating_matrix_test(user_seq, num_users, num_items):
                # three lists are used to construct sparse matrix
                row = []
                col = []
                data = []
                for user_id, item_list in enumerate(user_seq):
                    for item in item_list:  #
                        row.append(user_id)
                        col.append(item)
                        data.append(1)
    
                row = np.array(row)
                col = np.array(col)
                data = np.array(data)
                rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
    
                return rating_matrix
            self.args.user, self.args.item = trnMat[0].shape
            self.trnMat=generate_rating_matrix_test(self.sequence,self.args.user, self.args.item)
            # self.trnMat=trnMat[0]
            self.subMat = trnMat[1]
            self.timeMat = trnMat[2]
            print("trnMat",trnMat[0],trnMat[1],trnMat[2])
            self.tstInt = tstInt
            self.tstUsrs = tstUsrs
            self.prepareGlobalData()
        else:
            self.prepare(self.predir)
            self.trans_data()
            self.sequence = list(self.sequence)
            self.prepareGlobalData()

    def timeProcess(self,trnMats):
        mi = 1e16
        ma = 0
        for i in range(len(trnMats)):
            minn = np.min(trnMats[i].data)
            maxx = np.max(trnMats[i].data)
            mi = min(mi, minn)
            ma = max(ma, maxx)
        maxTime = 0
        for i in range(len(trnMats)):
            newData = ((trnMats[i].data - mi) // (3600*24*self.args.slot)).astype(np.int32)
            maxTime = max(np.max(newData), maxTime)
            trnMats[i] = csr_matrix((newData, trnMats[i].indices, trnMats[i].indptr), shape=trnMats[i].shape)
        print('MAX TIME',mi,ma, maxTime)
        return trnMats, maxTime + 1
    
    def prepareGlobalData(self):
        def tran_to_sym(R):
            adj_mat = sp.dok_matrix((self.args.user + self.args.item, self.args.user + self.args.item), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = R.tolil()
            adj_mat[:self.args.user, self.args.user:] = R
            adj_mat[self.args.user:, :self.args.user] = R.T
            adj_mat = adj_mat.tocsr()
            return (adj_mat+sp.eye(adj_mat.shape[0]))
            

        # adj = self.subMat
        self.maxTime=1
        # self.subMat,self.maxTime=self.timeProcess(self.subMat)
        print(self.subMat[0],self.subMat[-1])

        self.item_with_pop=[]
    def sampleLargeGraph(self, pckUsrs, pckItms=None, sampDepth=2, sampNum=None, preSamp=False):
        if sampNum is None:
            sampNum = self.args.graphSampleN
        adj = self.adj
        tpadj = self.tpadj
        def makeMask(nodes, size):
            mask = np.ones(size)
            if not nodes is None:
                mask[nodes] = 0.0
            return mask

        def updateBdgt(adj, nodes):
            if nodes is None:
                return 0
            tembat = 1000
            ret = 0
            for i in range(int(np.ceil(len(nodes) / tembat))):
                st = tembat * i
                ed = min((i+1) * tembat, len(nodes))
                temNodes = nodes[st: ed]
                ret += np.sum(adj[temNodes], axis=0)
            return ret

        def sample(budget, mask, sampNum):
            score = (mask * np.reshape(np.array(budget), [-1])) ** 2
            norm = np.sum(score)
            if norm == 0:
                return np.random.choice(len(score), 1), sampNum - 1
            score = list(score / norm)
            arrScore = np.array(score)
            posNum = np.sum(np.array(score)!=0)
            if posNum < sampNum:
                pckNodes1 = np.squeeze(np.argwhere(arrScore!=0))
                # pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
                # pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
                pckNodes = pckNodes1
            else:
                pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
            return pckNodes, max(sampNum - posNum, 0)

        def constructData(usrs, itms):
            adj = self.trnMat
            pckU = adj[usrs]
            tpPckI = transpose(pckU)[itms]
            pckTpAdj = tpPckI
            pckAdj = transpose(tpPckI)
            return pckAdj, pckTpAdj, usrs, itms

        usrMask = makeMask(pckUsrs, adj.shape[0])
        itmMask = makeMask(pckItms, adj.shape[1])
        itmBdgt = updateBdgt(adj, pckUsrs)
        if pckItms is None:
            pckItms, _ = sample(itmBdgt, itmMask, len(pckUsrs))
            itmMask = itmMask * makeMask(pckItms, adj.shape[1])
        usrBdgt = updateBdgt(tpadj, pckItms)
        uSampRes = 0
        iSampRes = 0
        for i in range(sampDepth + 1):
            uSamp = uSampRes + (sampNum if i < sampDepth else 0)
            iSamp = iSampRes + (sampNum if i < sampDepth else 0)
            newUsrs, uSampRes = sample(usrBdgt, usrMask, uSamp)
            usrMask = usrMask * makeMask(newUsrs, adj.shape[0])
            newItms, iSampRes = sample(itmBdgt, itmMask, iSamp)
            itmMask = itmMask * makeMask(newItms, adj.shape[1])
            if i == sampDepth or i == sampDepth and uSampRes == 0 and iSampRes == 0:
                break
            usrBdgt += updateBdgt(tpadj, newItms)
            itmBdgt += updateBdgt(adj, newUsrs)
        usrs = np.reshape(np.argwhere(usrMask==0), [-1])
        itms = np.reshape(np.argwhere(itmMask==0), [-1])
        return constructData(usrs, itms)

    def prepare(self, path):
        # 处理train
        # 获得interaction时间戳矩阵，interaction[usr][itm].extend([timeStamp])
        users, items = [], []
        maxx, minn = -1, 1 << 30
        with open(path + 'train.csv', 'r', encoding='utf=8') as file:
            # 验证用户id的正确性，并获得用户数量num_users和物品数量num_items
            reader = csv.reader(file)
            sign = True
            for row in reader:
                if sign:
                    print(row)
                    sign = False
                else:
                    tmp = row[0].split('\t')
                    index_users = int(tmp[0]) - 1
                    index_items = int(tmp[1]) - 1
                    timestamp = int(tmp[2])
                    maxx = max(maxx, timestamp)
                    minn = min(minn, timestamp)
                    users.append(index_users)
                    items.append(index_items)
            num_users = max(users) + 1
            num_items = max(items) + 1
            assert set(range(num_users)) == set(users)
            # 获得用户数量和物品数量
            self.args.user, self.args.item = num_users, num_items
            # 获得时间戳的最大值和最小值
            self.args.maxx, self.args.minn = maxx, minn

            file.seek(0)
            reader = csv.reader(file)
            interaction = [dict() for _ in range(num_users)]
            sign = True
            for row in reader:
                if sign:
                    print(row)
                    sign = False
                else:
                    tmp = row[0].split('\t')
                    index_user = int(tmp[0]) - 1
                    index_item = int(tmp[1]) - 1
                    # train_data[index].append([int(tmp[1]), int(tmp[2])])
                    if interaction[index_user].get(index_item, -1) == -1:
                        interaction[index_user][index_item] = []
                    interaction[index_user][index_item].append(int(tmp[2]))

        # 形状： (num_users, dict(item_id: [timestamps]))
        self.interaction = interaction

        # 获得sequence矩阵，sequence[usr] = [item_id1, item_id2, ...]按时间戳升序排列
        sequence = [[] for _ in range(num_users)]
        for u in range(num_users):
            item_time_list = []
            for item in interaction[u]:
                for t in interaction[u][item]:
                    item_time_list.append((item, t))
            item_time_list = sorted(item_time_list, key=lambda x: x[1])
            sequence[u] = [item for item, t in item_time_list]
        self.sequence = np.array(sequence)

        # 处理dev
        with open(path + "dev.csv", 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = []
            sign = True
            for row in reader:
                if sign:
                    print(row)
                    sign = False
                else:
                    data.append((''.join(row)).split('\t'))
            for i in range(len(data)):
                data[i][0] = int(data[i][0]) - 1
                data[i][1] = int(data[i][1]) - 1
                data[i][2] = int(data[i][2])
                data[i][3] = (data[i][3][1:-1]).split(' ')
                data[i][3] = [int(x) for x in data[i][3]]
        # 形状： (num_dev_samples, 4)
        # 每个样本格式为 [user_id, item_id, timest amp, [negative_item_ids]]
        dev_data = data
        # 将dev数据转换为论文的形式，加到训练数据的后面
        # 能够保证测试数据不重复
        self.map_dev_neg = dict()
        for i in range(len(dev_data)):
            user_id, item_id = dev_data[i][0], dev_data[i][1]
            negative_item_ids = dev_data[i][3]
            self.map_dev_neg[user_id] = negative_item_ids
            self.sequence[user_id] = np.concatenate((self.sequence[user_id], np.array([item_id])))

        # 处理test
        with open(path + "test.csv", 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = []
            sign = True
            for row in reader:
                if sign:
                    print(row)
                    sign = False
                else:
                    data.append((''.join(row)).split('\t'))
            for i in range(len(data)):
                data[i][0] = int(data[i][0]) - 1
                data[i][1] = int(data[i][1]) - 1
                data[i][2] = int(data[i][2])
                data[i][3] = (data[i][3][1:-1]).split(' ')
                data[i][3] = [int(x) for x in data[i][3]]
        data = np.array(data)
        # 形状： (num_test_samples, 3)
        # 每个样本格式为 [user_id, item_id, timestamp, [negative_item_ids]]
        test_data = data

        self.tstInt = test_data[:, 1].astype(np.int32)
        self.tstUsrs = test_data[:, 0].astype(np.int32)
        self.test_dict = dict()
        for i in range(len(test_data)):
            user_id = test_data[i][0] + 1
            negative_item_ids = test_data[i][3]
            self.test_dict[user_id] = negative_item_ids

    def trans_data(self):
        # trnMat: 记录用户交互的物品，二维矩阵，对应位置为1
        def generate_rating_matrix_test(user_seq, num_users, num_items):
            # three lists are used to construct sparse matrix
            row = []
            col = []
            data = []
            for u in range(len(user_seq)):
                items = user_seq[u].keys()
                for item in items:
                    row.append(u)
                    col.append(item)
                    data.append(1.0)

            row = np.array(row)
            col = np.array(col)
            data = np.array(data)
            rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
            return rating_matrix
        self.trnMat = generate_rating_matrix_test(self.interaction, self.args.user, self.args.item)

        # subMat: 划分子图，记录用户交互的物品，二维矩阵，对应位置为时间戳，可以传入time.locatetime()进行转换为可读的时间
        def trans_sub(interaction, usrnum, itmnum, gragh_num, maxx, minn):
            interval = (maxx - minn) / gragh_num
            rcd = [[list(), list(), list()]]
            for i in range(gragh_num - 1):
                rcd.append([list(), list(), list()])
            timeMat = sp.dok_matrix((usrnum, itmnum), dtype=np.int)
            for usr in range(usrnum):
                if interaction[usr] == None:
                    continue
                data = interaction[usr]
                for col in data:
                    if data[col] != None:
                        for one_data in data[col]:
                            gragh_no = int(((one_data - minn) / interval))
                            if (gragh_no >= gragh_num):
                                print(one_data, gragh_no)
                                gragh_no = gragh_num - 1
                            # print(gragh_no,one_data)
                            if (len(rcd[gragh_no][0]) == 0 or rcd[gragh_no][0][-1] != usr or rcd[gragh_no][1][
                                -1] != col):
                                rcd[gragh_no][0].append(usr)
                                rcd[gragh_no][1].append(col)
                                rcd[gragh_no][2].append(one_data)
                                # rcd[gragh_no][0].append(usrnum+col)
                                # rcd[gragh_no][1].append(usr)
                                # rcd[gragh_no][2].append(1.0)
                                timeMat[usr, col] = gragh_no
            intMat = list()
            for i in range(gragh_num):
                intMat.append(csr_matrix((rcd[i][2], (rcd[i][0], rcd[i][1])), shape=(usrnum, itmnum)))
                # intMat.append(normalized_adj_single(csr_matrix((rcd[i][2], (rcd[i][0], rcd[i][1])), shape=(usrnum+itmnum, usrnum+itmnum))))+itmnum usrnum+
                print(intMat[i])
            return intMat, timeMat.tocsr()
        self.subMat, self.timeMat = trans_sub(self.interaction, self.args.user, self.args.item, self.args.graphNum, self.args.maxx, self.args.minn)

