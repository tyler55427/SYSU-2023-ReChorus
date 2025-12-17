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
    # --- 情况 1: 输入是 SciPy 稀疏矩阵 (csr_matrix, coo_matrix 等) ---
    if sp.issparse(mat):
        # 转为 COO 格式以便提取行/列索引
        coomat = sp.coo_matrix(mat)
        
        # 1. 提取索引和数据
        # PyTorch 要求 indices 形状为 [2, NNZ]
        indices = np.vstack((coomat.row, coomat.col))
        data = coomat.data.astype(np.float32)
        shape = list(coomat.shape)

        # 2. 归一化处理 (numpy 实现)
        if norm:
            row_sum = np.array(coomat.sum(1)).flatten()
            col_sum = np.array(coomat.sum(0)).flatten()
            
            rowD = np.power(row_sum + 1e-8, -0.5)
            colD = np.power(col_sum + 1e-8, -0.5)
            
            # data[k] = data[k] * rowD[row[k]] * colD[col[k]]
            data = data * rowD[indices[0]] * colD[indices[1]]

        # 3. Mask 处理
        if mask:
            spMask = (np.random.uniform(size=data.shape) > 0.5).astype(np.float32)
            data = data * spMask

        # 4. 处理空矩阵
        if indices.shape[1] == 0:
            indices = np.array([[0], [0]], dtype=np.int64)
            data = np.array([0.0], dtype=np.float32)

        # 5. 返回 PyTorch Tensor
        return torch.LongTensor(indices), torch.FloatTensor(data), shape

    # --- 情况 2: 输入已经是 PyTorch 稀疏张量 ---
    elif torch.is_tensor(mat) and mat.is_sparse:
        mat = mat.coalesce()
        indices = mat.indices()
        data = mat.values()
        shape = list(mat.shape)

        if norm:
            row_sum = torch.sparse.sum(mat, [1]).to_dense()
            col_sum = torch.sparse.sum(mat, [0]).to_dense()
            rowD = torch.pow(row_sum + 1e-8, -0.5)
            colD = torch.pow(col_sum + 1e-8, -0.5)
            data = data * rowD[indices[0]] * colD[indices[1]]

        if mask:
            spMask = (torch.rand(data.shape) > 0.5).float().to(data.device)
            data = data * spMask
            
        return indices, data, shape

    else:
        raise ValueError(f"transToLsts received unsupported type: {type(mat)}")

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

class AdditiveAttention_Pytorch(nn.Module):
    def __init__(self,query_vector_dim,candidate_vector_dim):
        super(AdditiveAttention_Pytorch, self).__init__()
        self.query_vector_dim=query_vector_dim
        self.candidate_vector_dim=candidate_vector_dim
        self.attention_query_vector = nn.Parameter(torch.FloatTensor(query_vector_dim,1).uniform_(-0.1,0.1))
        
    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        dense  = nn.Linear(self.candidate_vector_dim, self.query_vector_dim)(candidate_vector)
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(dense)
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.squeeze(torch.matmul( temp, self.attention_query_vector),2),dim=1) #* 128
        # batch_size, 1, candidate_size * batch_size, candidate_size, candidate_vector_dim =
        # batch_size, candidate_vector_dim
        target =torch.squeeze( torch.matmul(torch.unsqueeze(candidate_weights,1),candidate_vector),1)
        #target = tf.multiply(candidate_weights,candidate_vector)
        return target
 
# class ScaledDotProductAttention(object):
#     def __init__(self, d_k):
#         self.d_k = d_k
    
#     def attention(self, Q, K, V, attn_mask=None):
#         with tf.name_scope('scaled_attention'): 
#             # batch_size,head_num, candidate_num, candidate_num
#             scores = tf.matmul(Q, tf.transpose(K,perm=[0,1,3,2])) / np.sqrt(self.d_k)
#             scores = tf.exp(scores)
#             if attn_mask is not None:
#                 scores = scores * attn_mask
#             # batch_size,head_num, candidate_num, 1
#             attn = scores / (tf.expand_dims(tf.reduce_sum(scores, axis=-1),-1) + 1e-8) # 归一化
#             context = tf.matmul(attn, V)
#             return context, attn
        
class ScaledDotProductAttention_Pytorch(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention_Pytorch, self).__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, attn_mask=None):
        # batch_size,head_num, candidate_num, candidate_num
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        # batch_size,head_num, candidate_num, 1
        attn = scores / (torch.unsqueeze(torch.sum(scores, dim=-1),-1) + 1e-8) # 归一化
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadSelfAttention_Pytorch(nn.Module):
    def __init__(self, d_model, num_attention_heads):
        super(MultiHeadSelfAttention_Pytorch, self).__init__()
        self.d_model = d_model # embedding_size
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads #16
        self.d_v = d_model // num_attention_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K=None, V=None, length=None):
        """
        Q:batch_size,candidate_num,embedding_size
        return : batch_size,candidate_num,embedding_size
        """
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.shape[0]
        W_Q = self.W_Q(Q)
        # batch_size, candidate_num, num_attention_heads,d_k  ;;divide into groups whose num is num_attention_heads
        # batch_size, num_attention_heads, candidate_num,d_k
        q_s = W_Q.view(batch_size, -1, self.num_attention_heads,self.d_k).permute(0,2,1,3)
        W_K = self.W_K(K)
        k_s = W_K.view(batch_size, -1, self.num_attention_heads,self.d_k).permute(0,2,1,3)
        W_V = self.W_V(V)
        v_s = W_V.view(batch_size, -1, self.num_attention_heads,self.d_v).permute(0,2,1,3)
        # batch_size,head_num, candidate_num, d_k
        context, attn = ScaledDotProductAttention_Pytorch(self.d_k).forward(q_s, k_s, v_s)#,attn_mask)
        # batch_size,candidate_num,embedding_size
        context= context.permute(0,2,1,3).contiguous().view(batch_size, -1, self.num_attention_heads*self.d_v)
        return context
