#coding=utf-8
import torch
from torch import nn, einsum
import math

from einops import rearrange,repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class CoPE(nn.Module):
    def __init__ (self, C, T=None) :
        '''
            CoPE实现
        :param C: 每个头的通道维度
        :param T:
        '''
        super().__init__()
        self.T = T if(T is not None) else C
        self.pos_emb = nn.parameter.Parameter(
            torch.zeros(1, C, self.T)
        )

    def forward (self, q, qk) :
        '''
            CoPE的思想是，融合上下文寻址和位置寻址。
            1. 通过qiT @ kj来产生上下文位置
            2. 利用传统的RPE方式，将通过上下文位置当做特征位置进行编码
        :param q: (B*H,N,C) or (B,H,N,C)
        :param qk: (B*H,N,N) or (B,H,N,N), q与k相乘的结果，不包含softmax部分
        :return: E: (B*H,N,N) or (B,H,N,N)
        '''
        '''
            q(B,H,N,C) @ kT(B,H,C,N) = qk(B,H,N,N)
            q中每一个行向量与kT中的列向量相乘，因此公式表达是qiT @ kj (qi,kj都表示的列向量)
            对上下文寻址，进行门限取值,取值范围为（0,1），取值越大，意味着权重越大
            这句代码，对应了公式(3), gij = sigmid(qiT @ ki)
            G(B,H,N,N)
        '''
        G = torch.sigmoid(qk)
        '''
            [B,N,N]沿着最后一个维度进行翻转特征，然后对最后一维度进行累计求和，最后再沿着最后一个维度将特征翻转刚回来
            [a,b,c] -> [c,b,a] -> [a+b+c,b+c,a] -> [a,b+c,a+b+c]
            这句代码，对应了公式(4), pij = sum{k=j ~ i}(gik)
            P(B,H,N,N)
        '''
        P = G.flip(-1).cumsum(dim=-1).flip(-1)
        P = P.clamp(max=self.T - 1)
        '''
            整型编码插值
            由于sigmod的原因，CoPE不能像传统的RPE一样，利用可学习的编码层学习位置信息
            因此，使用一种简单的整型向量插值方法，来融合可学习的编码特征
            以下代码对应公式(5)
            同时，公式(9)采用了一种更高效的实现
            E(B,H,N,N)
        '''
        P_ceil = P.ceil().long()
        P_floor = P.floor().long()
        # (B,H,N,C) @ (1,C,T) = (B,H,N,T)
        E = torch.matmul(q, self.pos_emb) # eij
        E_ceil = E.gather(-1, P_ceil)
        E_floor = E.gather(-1, P_floor)
        P_P_floor = P - P_floor
        #E = (P - P_floor) * E_cell + (1 - P + P_floor) * E_floor
        E = P_P_floor * E_ceil + (1 - P_P_floor) * E_floor
        return E


class Attention(nn.Module):

    def __init__(
            self,
            Q_dim,
            KV_dim = None,
            heads = 8,
            dim_head = 64
    ):
        super().__init__()
        inner_dim = dim_head * heads
        KV_dim = default(KV_dim, Q_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(Q_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(KV_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, Q_dim)

        self.cope = CoPE(dim_head)

    def forward(self, x, kv = None, mask=None):

        h = self.heads
        q = self.to_q(x)
        kv = default(kv, x)
        k,v = self.to_kv(kv).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        pe = self.cope(q, sim)
        sim = sim + pe

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return out

if __name__ == '__main__':

    q = torch.ones(size=(2,128,64))
    a = Attention(Q_dim=64)
    x = a(q)
    print(x.shape)