import math
from functools import partial
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction,BertConfig, BertModel,AutoConfig

class NystromAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 64
        self.num_head = 12
        self.num_landmarks = 512
        self.seq_len = 64
        self.init_option = "original"
        self.conv = nn.Conv2d(
            in_channels = self.num_head, out_channels = self.num_head,
            kernel_size = (33, 1), padding = (33 // 2, 0),
            bias = False,
            groups = self.num_head)

    def forward(self, Q, K, V):

        Q = Q / math.sqrt(math.sqrt(self.head_dim))
        K = K / math.sqrt(math.sqrt(self.head_dim))

        Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
        K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

        kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
        kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
        kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) - 1e9, dim = -1)
        X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        X += self.conv(V)

        return X
    
    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0. 
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.dim = 1024
        self.head_dim = 64
        self.num_head = 12
        self.attn_type = "nystrom"

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = NystromAttention()
        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X1, X2, return_QKV = False):

        Q = self.split_heads(self.W_q(X1))
        K = self.split_heads(self.W_k(X2))
        V = self.split_heads(self.W_v(X2))
        with torch.cuda.amp.autocast(enabled = False):
            attn_out = self.attn(Q.float(), K.float(), V.float())
        attn_out = self.combine_heads(attn_out)
        out = self.ff(attn_out)

        if return_QKV:
            return out, (Q, K, V)
        else:
            return out
        
    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), 1, self.num_head * self.head_dim)
        return X
        
    def split_heads(self, X):
        X = X.reshape(X.size(0), 1, self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'

class CrossAttention3D(nn.Module):
    def __init__(self, query_dim=512, context_dim=512, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = max(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        a = self.heads
        # b, c, n, h, w = x.shape  # 3D input: batch, time, channel, height, width
        # x = rearrange(x, 'b c n h w -> b (n h w) c ')
        q = self.to_q(x).unsqueeze(1)
        context = context if context is not None else x
        k = self.to_k(context).unsqueeze(1)
        v = self.to_v(context).unsqueeze(1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=a), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=a)
        out = self.to_out(out)
        # out = self.to_out(out).view(b, c, -1, h, w)
        return out.squeeze(1)


class Fusion_survival(nn.Module):
    def __init__(self,              
                 ct_encoder=None,
                 num_classes=1, 
                 vit = 'base',                
                 embed_dim = 1024,     
                 queue_size = 57600,
                 momentum = 0.995,
                 ):
        
        super().__init__()
        # self.visual_encoder = PatchGCN_Surv() 
        if vit=='base':
            vision_width = 512
        elif vit=='large':
            vision_width = 1024
        self.ct_encoder = ct_encoder

        self.ct_vision_proj = nn.Linear(1024, embed_dim)
        self.text_proj = nn.Linear(4096, embed_dim)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   

        self.last_linear = nn.Linear(1024,num_classes)
        
    def forward(self, ct, rois, ct_text):   
        ct_embeds_m = self.ct_encoder(ct, rois)
        ct_text = ct_text.squeeze(1).cuda().float()
        ct_text_feat = self.text_proj(ct_text)
        ct_feat_m = self.ct_vision_proj(ct_embeds_m)

        all_ct_features = 0.1 * ct_text_feat + 0.9 * ct_feat_m

        return self.last_linear(all_ct_features)
