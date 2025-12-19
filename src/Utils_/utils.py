import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

def transpose(mat):
    coomat = sp.coo_matrix(mat)
    return sp.csr_matrix(coomat.transpose())

def negSamp(temLabel, sampSize, nodeNum,trnPos, item_with_pop):
    negset = [None] * sampSize
    cur = 0
    # print(trnPos)
    while cur < sampSize:

        # rdmItm = random.choice(item_with_pop)
        # rdmItm = np.random.choice(sequence[rdmItm],1)
        rdmItm = np.random.choice(nodeNum)
        # if rdmItm not in temLabel and rdmItm != trnPos:
        if temLabel[rdmItm] == 0 and rdmItm not in trnPos:
            negset[cur] = rdmItm
            cur += 1
    return negset

def posSamp(user_sequence,sampleNum):
    indexs=np.random.choice(np.array(range(len(user_sequence))),sampleNum)
    # print(indexs)
    return user_sequence[indexs.sort()]

def transToLsts(mat, mask=False, norm=False):
    """将稀疏矩阵转换为列表形式"""
    shape = [mat.shape[0], mat.shape[1]]
    coomat = sp.coo_matrix(mat)
    indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int64)
    data = coomat.data.astype(np.float32)

    if norm:
        rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
        colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
        for i in range(len(data)):
            row = indices[i, 0]
            col = indices[i, 1]
            data[i] = data[i] * rowD[row] * colD[col]

    if mask:
        spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
        data = data * spMask

    if indices.shape[0] == 0:
        indices = np.array([[0, 0]], dtype=np.int64)
        data = np.array([0.0], dtype=np.float32)
    return indices, data, shape
#########################################

paramId = 0
biasDefault = False
params = {}
# regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
    global paramId
    paramId += 1
    return paramId

def setIta(ITA):
    ita = ITA

def setBiasDefault(val):
    global biasDefault
    biasDefault = val

def getParam(name):
    return params[name]


class BNLayer(nn.Module):
    def __init__(self, dim, ita=0.2):
        super(BNLayer, self).__init__()
        self.ita = ita
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(0)
            batch_var = x.var(0, unbiased=False)
            x_normed = (x - batch_mean) / torch.sqrt(batch_var + 1e-8)
            self.running_mean = self.ita * batch_mean + (1 - self.ita) * self.running_mean
            self.running_var = self.ita * batch_var + (1 - self.ita) * self.running_var
        else:
            x_normed = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        out = self.scale * x_normed + self.shift
        return out

# def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
#     global params
#     global regParams
#     global leaky
#     inDim = inp.get_shape()[1]
#     temName = name if name!=None else 'defaultParamName%d'%getParamId()
#     W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
#     if dropout != None:
#         ret = tf.nn.dropout(inp, rate=dropout) @ W
#     else:
#         ret = inp @ W
#     if useBias:
#         ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initializer=biasInitializer)
#     if useBN:
#         ret = BN(ret)
#     if activation != None:
#         ret = Activate(ret, activation)
#     return ret

class FC_Layer(nn.Module):
    def __init__(self, inDim, outDim, useBias=False, activation=None, reg=False, reuse=None, name=None, useBN=False, dropout=None, initializer='xavier', biasReg=False, biasInitializer='zeros'):
        super(FC_Layer, self).__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.useBias = useBias
        self.activation = activation
        self.reg = reg
        self.useBN = useBN
        self.dropout = dropout
        self.initializer = initializer
        self.biasReg = biasReg
        self.biasInitializer = biasInitializer

        self.W = nn.Parameter(torch.empty(inDim, outDim))
        nn.init.xavier_uniform_(self.W)
        
        if self.useBias:
            self.bias = nn.Parameter(torch.zeros(outDim))
        if self.useBN:
            self.bn = BNLayer(outDim)

    def forward(self, inp):
        if self.dropout != None:
            ret = F.dropout(inp, p=self.dropout) @ self.W
        else:
            ret = inp @ self.W
        if self.useBias:
            ret = ret + self.bias
        if self.useBN:
            ret = self.bn(ret)
        if self.activation != None:
            ret = ActivateHelp(ret, self.activation)
        return ret

# def Bias(data, name=None, reg=False, reuse=False, initializer='zeros'):
#     inDim = data.get_shape()[-1]
#     temName = name if name!=None else 'defaultParamName%d'%getParamId()
#     temBiasName = temName + 'Bias'
#     bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer=initializer, reuse=reuse)
#     if reg:
#         regParams[temBiasName] = bias
#     return data + bias

class Bias_Layer(nn.Module):
    def __init__(self, dim, reg=False, initializer='zeros'):
        super(Bias_Layer, self).__init__()
        self.dim = dim
        self.reg = reg
        self.initializer = initializer
        self.bias = nn.Parameter(self.init_weight([dim], initializer))

    def forward(self, data):
        return data + self.bias

def ActivateHelp(data, method):
    if method == 'relu':
        ret = nn.ReLU()(data)
    elif method == 'sigmoid':
        ret = nn.Sigmoid()(data)
    elif method == 'tanh':
        ret = nn.Tanh()(data)
    elif method == 'softmax':
        ret = nn.Softmax(dim=-1)(data)
    elif method == 'leakyRelu':
        ret = nn.LeakyReLU(negative_slope=leaky)(data)
    elif method == 'twoWayLeakyRelu6':
        temMask = (data > 6.0).float()
        ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * torch.max(leaky * data, data)
    elif method == '-1relu':
        ret = torch.max(-1.0 * torch.ones_like(data), data)
    elif method == 'relu6':
        ret = torch.clamp(data, min=0.0, max=6.0)
    elif method == 'relu3':
        ret = torch.clamp(data, min=0.0, max=3.0)
    else:
        raise Exception('Error Activation Function')
    return ret

# def Activate(data, method, useBN=False):
#     global leaky
#     if useBN:
#         ret = BN(data)
#     else:
#         ret = data
#     ret = ActivateHelp(ret, method)
#     return ret

class Activate(nn.Module):
    def __init__(self, dim, method, useBN=False):
        super(Activate, self).__init__()
        self.method = method
        self.useBN = useBN
        if self.useBN:
            self.bn = BNLayer(dim)  # dim needs to be defined or passed

    def forward(self, data):
        if self.useBN:
            ret = self.bn(data)
        else:
            ret = data
        ret = ActivateHelp(ret, self.method)
        return ret

def RegularizePytorch(regParams, names=None, method='L2'):
    ret = 0
    for param in regParams:
        ret += torch.sum(torch.square(param))
    return ret

def DropoutPytorch(data, rate):
    if rate == None:
        return data
    else:
        return F.dropout(data, p=rate)

class SelfAttentionPytorch(nn.Module):
    def __init__(self, inpDim, numHeads):
        super(SelfAttentionPytorch, self).__init__()
        self.inpDim = inpDim
        self.numHeads = numHeads
        self.Q = nn.Parameter(torch.randn(inpDim, inpDim))
        self.K = nn.Parameter(torch.randn(inpDim, inpDim))
        self.V = nn.Parameter(torch.randn(inpDim, inpDim))

    def forward(self, localReps, number):
        rspReps = torch.reshape(torch.stack(localReps, dim=1), [-1, self.inpDim])
        q = torch.reshape(rspReps @ self.Q, [-1, number, 1, self.numHeads, self.inpDim // self.numHeads])
        k = torch.reshape(rspReps @ self.K, [-1, 1, number, self.numHeads, self.inpDim // self.numHeads])
        v = torch.reshape(rspReps @ self.V, [-1, 1, number, self.numHeads, self.inpDim // self.numHeads])
        att = F.softmax(torch.sum(q * k, dim=-1, keepdim=True) / torch.sqrt(torch.tensor(self.inpDim / self.numHeads, dtype=torch.float32)), dim=2)
        attval = torch.reshape(torch.sum(att * v, dim=2), [-1, number, self.inpDim])
        rets = [None] * number
        for i in range(number):
            tem1 = torch.reshape(attval[:, i:i + 1, :], [-1, self.inpDim])
            rets[i] = tem1 + localReps[i]
        return rets


def normalize_adj(mat):
    """
    对称归一化: D^-0.5 * A * D^-0.5
    """
    rowsum = np.array(mat.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-8, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    colsum = np.array(mat.sum(0)).flatten()
    d_inv_sqrt_col = np.power(colsum + 1e-8, -0.5)
    d_inv_sqrt_col[np.isinf(d_inv_sqrt_col)] = 0.
    d_mat_inv_sqrt_col = sp.diags(d_inv_sqrt_col)

    return d_mat_inv_sqrt @ mat @ d_mat_inv_sqrt_col

def trans_to_cuda_sparse(mat):
    mat = normalize_adj(mat)
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.tensor(values, dtype=torch.int32)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.query_vector_dim = query_vector_dim
        self.candidate_vector_dim = candidate_vector_dim
        self.dense = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(torch.empty(query_vector_dim, 1).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector):
        """
        candidate_vector: [batch_size, candidate_size, candidate_vector_dim]
        return: [batch_size, candidate_vector_dim]
        """
        dense = self.dense(candidate_vector)
        temp = torch.tanh(dense)
        candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector).squeeze(-1), dim=1)
        target = torch.matmul(candidate_weights.unsqueeze(1), candidate_vector).squeeze(1)
        return target
        
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.scale = 1.0 / np.sqrt(d_k)

    def forward(self, Q, K, V, attn_mask=None):
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]
        # 使用更高效的计算方式，避免创建过多中间变量
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)  # 使用softmax替代exp+归一化，更稳定且高效
        del scores  # 及时释放
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制 - 与TF实现保持一致"""
    def __init__(self, d_model, num_attention_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads
        
        # TF使用分开的W_Q, W_K, W_V (tf.layers.dense)
        # 每个dense层独立初始化xavier_uniform
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / np.sqrt(self.d_k)
        
        # Xavier初始化 - 与TF的tf.contrib.layers.xavier_initializer一致
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)

    def forward(self, Q, K=None, V=None):
        """
        Q: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len, d_model]
        """
        if K is None:
            K = Q
        if V is None:
            V = Q
            
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # 分别计算Q, K, V投影
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads, self.d_v).transpose(1, 2)
        
        # TF原始实现: scores = exp(QK^T/sqrt(d_k)), attn = scores / sum(scores)
        # 这与softmax数学等价，但使用softmax更稳定
        scores = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scale
        
        # TF使用 exp + 手动归一化，为了数值一致性我们也这样做
        scores_exp = torch.exp(scores)
        attn = scores_exp / (scores_exp.sum(dim=-1, keepdim=True) + 1e-8)
        del scores, scores_exp
        
        context = torch.matmul(attn, v_s)
        del attn, q_s, k_s, v_s
        
        # Concat heads: [batch, heads, seq, d_k] -> [batch, seq, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return context