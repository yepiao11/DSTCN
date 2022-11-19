# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:21:28 2018

@author: gk
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d

"""
Created on Mon Aug 20 13:20:23 2018

@author: gk
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d
import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""
class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_0 \cdot \mathbf{x}_i +
        \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 bias=True,
                 **kwargs):
        super(RGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations #3
        self.num_bases = num_bases  #3

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.basis)
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.root)

    def forward(self, x, edge_index, edge_attr, edge_norm=None): #inputs=[640,2],  edge_index:torch.Size([2, 25712]),edge_attr=torch.Size([25712,3])
        """"""
        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr, edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_attr, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        out = torch.einsum('bi,rio->bro', x_j, w)
        out = (out * edge_attr.unsqueeze(2)).sum(dim=1)
        # print(out.size(), edge_attr.unsqueeze(2).size())
        # out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if x is None:
            out = aggr_out + self.root
        else:
            out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)

class KStepRGCN(nn.Module):
    """docstring for KStepRGCN"""

    def __init__(
            self,
            in_channels,  #80
            out_channels,  #256
            num_relations,   #3
            num_bases,  #3
            K, #1
            bias, #False
    ):
        super(KStepRGCN, self).__init__()
        self.in_channels = in_channels   #80
        self.out_channels = out_channels  #256
        self.num_relations = num_relations    #3
        self.num_bases = num_bases #3
        self.K = K  #1
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(in_channels,  #80
                     out_channels,  #256
                     num_relations,  #3
                     num_bases,  #3
                     bias)  #False
        ] + [
            RGCNConv(out_channels,   #256*3
                     out_channels,  #256*3
                     num_relations,   #3
                     num_bases,   #3
                     bias) for _ in range(self.K - 1)
        ])


    def forward(self, x, edge_index, edge_attr):
        edge_index=edge_index.to(x.device)
        edge_attr=edge_attr.to(x.device)
        # print('###')
        # print(x.shape)#torch.Size([1280, 80])
        # print(edge_index.shape)#torch.Size([2, 3214])
        # print(edge_attr.shape)#torch.Size([3214, 3])
        for i in range(self.K):
            x = self.rgcn_layers[i](x=x,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    edge_norm=None)
            # not final layer, add relu
            if i != self.K - 1:
                x = torch.relu(x)
        # print('1234')
        # print(x.shape)  #torch.Size([1280, 256])
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):#torch.Size([16, 197, 384])
        # print('Mlp')
        # print(x.shape)
        x = self.fc1(x)#torch.Size([16, 197, 1536])
        # print(x.shape)
        x = self.act(x)#torch.Size([16, 197, 1536])
        # print(x.shape)
        x = self.drop(x)#torch.Size([16, 197, 1536])
        # print('xxxxx')
        # print(x.shape)
        x = self.fc2(x)#torch.Size([16, 197, 384])
        # print(x.shape)
        x = self.drop(x)#torch.Size([16, 197, 384])
        # print(x.shape)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):#torch.Size([16, 197, 384])
        # print('Attention')
        # print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#torch.Size([3, 16, 6, 197, 64])
        # print(qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print('q,k')
        # print(q.shape)#torch.Size([16, 6, 197, 64])

        attn = (q @ k.transpose(-2, -1)) * self.scale#torch.Size([16, 6, 197, 197])
        # print(attn.shape)
        attn = attn.softmax(dim=-1)#torch.Size([16, 6, 197, 197])
        # print(attn.shape)
        attn = self.attn_drop(attn)#torch.Size([16, 6, 197, 197])
        # print('attn')
        # print(attn.shape)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#torch.Size([16, 197, 384])
        # print(x.shape)
        x = self.proj(x)#torch.Size([16, 197, 384])
        # print(x.shape)
        x = self.proj_drop(x)#torch.Size([16, 197, 384])
        # print('567890-')
        # print(x.shape)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):#torch.Size([16, 197, 384])
        # print('Block')
        # print(x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))#torch.Size([16, 197, 384])
        # print('x1')
        # print(x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))#torch.Size([16, 197, 384])
        # print('x2')
        # print(x.shape)
        return x



class GRU(nn.Module):
    def __init__(self,device):
        super(GRU, self).__init__()
        self.device=device
        c_in=69
        c_out=256
        self.node_num=69
        self.gru = nn.GRU(c_in, c_out, batch_first=True)  # b*n,l,c
        self.c_out = c_out
        self.bn = BatchNorm2d(c_in, affine=False)

    def forward(self, x1):## torch.Size([16, 30, 7]),# torch.Size([16, 30, 108])

        # print('h')
        shape = x1.shape#torch.Size([16, 80, 80, 30])
        B=shape[0]
        # print(x.shape)
        h = Variable(torch.zeros(1, shape[0] * shape[2], self.c_out,dtype=x1.dtype,device=x1.device))#torch.Size([1, 1280, 256])
        # print(h.shape)
        hidden = h#torch.Size([1, 1280, 256])
        # print(hidden.shape)
        x = x1.permute(0, 2, 3, 1).contiguous().view(shape[0] * shape[2], shape[3], shape[1])#torch.Size([1280, 6, 80]),torch.Size([1280, 30, 256])
        # print(x.shape)
        x, hidden = self.gru(x, hidden)#torch.Size([1280, 6, 256]),torch.Size([1280, 30, 256])
        # print('68790-')
        # print(x.shape)
        x = x.view(shape[0], shape[2], shape[3], self.c_out).permute(0, 3, 1, 2).contiguous()#torch.Size([16, 256, 80, 30])
        return x

class Model(nn.Module):
    def __init__(self, device,numT,edge_index,edge_attr):
        super(Model, self).__init__()
        c_in=69
        num_nodes=69
        self.device=device
        self.bn = BatchNorm2d(c_in, affine=False)
        C = 256  # C是channel
        D = 256  # D是hidden dimension
        self.GRU=GRU(device)
        self.with_vertical=True
        self.with_horizontal=True
        self.union = "cat"

        self.rnn_v=nn.LSTM(numT,D,num_layers=1,batch_first=True,bias=True,bidirectional=True)
        self.rnn_h = nn.LSTM(numT, D, num_layers=1, batch_first=True, bias=True, bidirectional=True)

        self.fc = nn.Linear(4*D, D)

        # self.f1 = nn.Linear(30,D)
        self.f2 = nn.Linear(D, numT)
        self.D=D

        #加入transformer
        self.conv = nn.Conv2d(numT, 256, kernel_size=7, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.trans_patch_conv = nn.Conv2d(256, 768, kernel_size=2, stride=2, padding=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        # trans_block
        embed_dim = 768
        num_heads = 6
        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.0
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
        patch_size = 2
        self.unconv1 = nn.ConvTranspose2d(768, 256, patch_size, patch_size)
        self.unconv2 = nn.ConvTranspose2d(256, 256, patch_size, patch_size)
        self.conv_shape = nn.Conv2d(256, 256, kernel_size=2, padding=1)
        # self.fc_1 = nn.Linear(80, 256)
        self.mlpatt = nn.Linear(2 * D, D)
        #额外加的
        self.edge_index=edge_index
        self.edge_attr=edge_attr
        self.numT=numT
        c_out = 69
        self.node_num = 69
        self.cheb_v = KStepRGCN(69,  # 80
                                c_out,  # 256
                                num_relations=3,  # 3
                                num_bases=3,  # 3
                                K=1,  # 1
                                bias=False)
        self.gcn_v_fc=nn.Linear(numT, 2*D)
        self.mlp_v = nn.Linear(4 * D, 2*D)
        self.cheb_h = KStepRGCN(69,  # 80
                                c_out,  # 256
                                num_relations=3,  # 3
                                num_bases=3,  # 3
                                K=1,  # 1
                                bias=False)
        self.gcn_h_fc = nn.Linear(numT, 2 * D)
        self.mlp_h = nn.Linear(4 * D, 2 * D)
        # self.register_buffer('my_buffer', self.edge_index)
        # self.register_buffer('my_buffer', self.edge_attr)
        self.fc_toweek = nn.Linear(numT*7, 768)
        self.fc_tohour = nn.Linear(numT*24 , 768)
        # self.W1 = nn.Parameter(torch.FloatTensor(289, embed_dim).to(device))
        # self.W2 = nn.Parameter(torch.FloatTensor(289, embed_dim).to(device))
        # self.simformer_1 = nn.Linear(numT, D)
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.simformer_1 = nn.Linear(numT, D)
        self.norm2 = nn.LayerNorm(D)



    def forward(self, x_f,external_toweek, external_tohour):
        # print('external_toweek')
        # print(external_toweek.shape)#torch.Size([16, 6, 7])
        # print(external_toweek[0][0])
        external_toweek=external_toweek.flatten(1)#torch.Size([16, 6*7])

        # print(external_toweek.shape)
        external_toweek = self.fc_toweek(F.relu(external_toweek))  # torch.Size([16, 7])--->torch.Size([16,768]
        # print(external_toweek.shape)
        B,E=external_toweek.shape
        external_toweek = external_toweek.expand(289,B,E)  # torch.Size([16,768]--->torch.Size([16,1,768]
        # print(external_toweek.shape)
        external_toweek=external_toweek.permute(1,0,2)#(16,289,768)
        # print(external_toweek.shape)
        # print('external_tohour')
        # print(external_toweek.shape)#torch.Size([16, 6, 108])

        external_tohour=external_tohour.flatten(1)#torch.Size([16, 108])
        # print(external_tohour[0])
        # print(external_tohour.shape)
        external_tohour = self.fc_tohour(F.relu(external_tohour))  # torch.Size([16, 7*T])--->torch.Size([16,768]
        # print(external_tohour.shape)
        external_tohour = external_tohour.expand(289,B,E)
        # print(external_tohour.shape)
        external_tohour=external_tohour.permute(1,0,2)
        x=x_f
        B,H,W,C=x.shape#B=16, H=80, W=80, C=256 

        if self.with_vertical:
            # print('vvvv')
            v = x.permute(0, 2, 1, 3)  # (B, H, W, C)--->(B,W,H,C),torch.Size([16, 80, 80, T])-->torch.Size([16, 80, 80, T])
            # print(v.shape)
            v1 = v.reshape(-1, H, C)  # (B,W,H,C)-->(B*W,H,C),torch.Size([16, T, 80, 80])-->torch.Size([16*80, 80, T])
            # print(v.shape)
            v, _ = self.rnn_v(v1)  # (B*W,H,2C),torch.Size([16*80, 80, 512])
            # print(v.shape)
            # v, _ = self.rnn_v1(v)# (B*W,H,2C),torch.Size([16*80, 80, 60])
            # print(v.shape)
            v = v.reshape(B, W, H, -1)  # (B*W,H,2C)-->(B,W,H,2C),torch.Size([16*80, 80, 2C])-->torch.Size([16,80, 80, 2C])
            v = v.permute(0, 2, 1, 3)  # (B,W,H,2C)--->(B,H,W,2C)#torch.Size([16,80, 80, 2C])-->torch.Size([16,80,80,2C])
            # print(v.shape)
            #加入图卷积去更新
            need_concat_v = []
            for i in range(self.numT):#12
                h_v = self.cheb_v(v1[:, :, i], edge_index=self.edge_index,
                                  edge_attr=self.edge_attr)  #torch.Size([16*80, 80, T])--->torch.Size([16*80, 80])
                # print(h_v.shape)#torch.Size([1280, 80])
                need_concat_v.append(h_v)
            gcn_v = torch.stack(need_concat_v, dim=-1)  # torch.Size([16*80, 80,T])
            # print('gcn_v')
            # print(gcn_v.shape)
            gcn_v = gcn_v.reshape(B, W, H, -1)#torch.Size([16, 80, 80, 12])
            # print(gcn_v.shape)
            gcn_v = gcn_v.permute(0, 2, 1, 3)
            gcn_v=self.gcn_v_fc(gcn_v)# torch.Size([16,80, 80,256*2])
            # print(gcn_v.shape)
            #融合双向lstm和gcn
            combine_v= torch.cat([v, gcn_v], dim=-1)  ##torch.Size([16, 80,80,512*2])
            # print('combine1')
            # print(combine_v.shape)
            v = v+v * torch.tanh(self.mlp_v(combine_v))#torch.Size([16, 80,80,512])
            # print(v.shape)
        if self.with_horizontal:
            # print('rtyuio')
            h1 = x.reshape(-1, W, C)#(B, H, W, C)-->(B*H, W, C),torch.Size([16, 80, 80, T])--->#torch.Size([16*80, 80, T])
            # print(h.shape)
            h, _ = self.rnn_h(h1)#(B*H, W, 2C),#torch.Size([16*80, 80, 512])
            # h, _ = self.rnn_h1(h)
            # print(h.shape)
            h = h.reshape(B, H, W, -1)#(B*H, W, 2C)--->(B,H, W, 2C),torch.Size([16*80, 80, 512])--->torch.Size([16,80,80, 512])
            # print(h.shape)
            # 加入图卷积去更新
            need_concat_h = []
            for i in range(self.numT):  # 12
                h_h = self.cheb_h(h1[:, :, i], edge_index=self.edge_index,
                                  edge_attr=self.edge_attr)  # torch.Size([16*80, 80, T])--->torch.Size([16*80, 80])
                # print(h_v.shape)#torch.Size([1280, 80])
                need_concat_h.append(h_h)
            gcn_h = torch.stack(need_concat_h, dim=-1)  # torch.Size([16*80, 80,T])
            # print('gcn_h')
            # print(gcn_h.shape)
            gcn_h= gcn_h.reshape(B, H,W, -1)  # torch.Size([16, 80, 80, 12])
            # print(gcn_h.shape)
            gcn_h = self.gcn_h_fc(gcn_h)  # torch.Size([16,80, 80,256*2])
            # print(gcn_h.shape)
            # 融合双向lstm和gcn
            combine_h = torch.cat([h, gcn_h], dim=-1)  ##torch.Size([16, 80,80,512*2])
            # print('combine')
            # print(combine_h.shape)
            h = h+h * torch.tanh(self.mlp_h(combine_h))  # torch.Size([16, 80,80,512])
            # print(h.shape)
        if self.with_vertical and self.with_horizontal:
            if self.union == "cat":
                x = torch.cat([v, h], dim=-1)#(B,H, W, 4C)#torch.Size([16, 80, 80, 1024])
                # print('cat')
                # print(x.shape)
            else:
                x = v + h
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h
        # print('76890')
        # print(x.shape)
        x_bilstm = self.fc(x)##torch.Size([16,80, 80, 4C])-->#torch.Size([16,80, 80, C]),torch.Size([16, 80, 80, 256])
        # print(x_bilstm.shape)
        #在这里我感觉可以进行一下空间transformer
        x1 = x_f.permute(0, 3, 1, 2)  #torch.Size([16, 6, 69, 69])
        # print('67890')
        # print(x1.shape)
        x = self.act1(self.bn1(self.conv(x1)))  # torch.Size([16,30,80,80])---->torch.Size([16, 256, 34, 34])
        # print('90-')
        # print(x.shape)
        x = self.trans_patch_conv(x).flatten(2).transpose(1, 2)  # torch.Size([16, 289, 768])
        # print(x.shape)
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # torch.Size([16, 1, 768])，torch.Size([16, 1, 768])
        # print(self.W1.shape)
        # print(self.W2.shape)
        x_t = x+F.relu(external_toweek+external_tohour)  #torch.Size([16, 400+2, 768])
        # print('xiio')
        # print(x_t.shape)
        x_t = self.trans_block(x_t)  # torch.Size([16, 291, 768]),讲x_st和全局加起来去做transformer
        # print('x_t')
        # print(x_t.shape)#torch.Size([16, 291, 768])
        # x_t = x_t[:,2 :, :]#torch.Size([16, 289, 768])
        # print(x_t.shape)
        x_t = x_t.view(-1, 17, 17, 768).permute(0, 3, 1, 2)  # torch.Size([16, 768, 17, 17])
        # print('7890')
        # print(x_t.shape)#torch.Size([16, 768, 14, 14]),torch.Size([16, 768, 20, 20])
        x_t = self.unconv1(x_t)  #torch.Size([16, 256, 68, 68])
        # print(x_t.shape)
        x_t = self.unconv2(x_t)
        # print(x_t.shape)
        x_t = self.conv_shape(x_t)  # torch.Size([16, 256, 69, 69])
        # print(x_t.shape)
        x_t = x_t.permute(0, 2, 3, 1)#torch.Size([16, 69, 69, 256])
        # print(x_t.shape)
        # print('simformer_1')
        # print(x_f.shape)
        # print(self.simformer_1(x_f).shape)
        x_t = self.simformer_1(x_f) + x_t
        # print(x_t.shape)
        # print('88889789')
        x_t = self.norm2(x_t)
        #融合双向lstm和全局transformer
        combine_hidden = torch.cat([x_bilstm,x_t], dim=-1)  ##torch.Size([16, 80,80,512])
        # print(combine_hidden.shape)
        next_hidden = x_bilstm * torch.tanh(self.mlpatt(combine_hidden))

        x = self.f2(next_hidden)  # torch.Size([16,80, 80, C])--->torch.Size([16, 69, 69, T])
        # print('7890-')
        # print(x.shape)
        # 加入GRU
        x = self.GRU(x)  ##torch.Size([16, 256, 80, T]),torch.Size([16, 256, 80, T])
        # print('jihuo')
        # print(x.shape)
        return x




class Net_block(torch.nn.Module):

    def __init__(self,device,edge_index,edge_attr):
        super(Net_block, self).__init__()
        self.bn = BatchNorm2d(69, affine=False)
        self.submodule=Model(device,6,edge_index,edge_attr)
        # self.DEVICE = DEVICE
        # self.to(DEVICE)
        D=256
        num_nodes=69
        self.conv1 = Conv2d(D, num_nodes, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv2 = Conv2d(D, num_nodes, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv3 = Conv2d(D, num_nodes, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv4 = Conv2d(D, num_nodes, kernel_size=(1, 2), padding=(0, 0),
                            stride=(1, 2), bias=True)


    def forward(self,x_list,external_list):
        # 加载data数据
        x_w, x_d, x_r = x_list[0], x_list[1], x_list[
            2]  # torch.Size([16, 80, 80, T]),torch.Size([16, 80, 80, T]),torch.Size([16, 80, 80, T])
        # print('ghj')
        # print(x_w.shape)#torch.Size([16, 69, 69, 2])
        # print(x_d.shape)#torch.Size([16, 69, 69, 1])
        # print(x_r.shape)#torch.Size([16, 69, 69, 3])
        ext_w_toweek, ext_w_tohour,ext_d_toweek, ext_d_tohour,ext_r_toweek, ext_r_tohour= \
            external_list[0], external_list[1],external_list[2], external_list[3],external_list[4], external_list[5]  # torch.Size([16, 6, 7]),torch.Size([16, 6, 108])
        # print('ext_w_toweek')
        # print(ext_w_toweek.shape)#torch.Size([16, 2, 7])
        # print(ext_w_tohour.shape)#torch.Size([16, 2, 24])
        # print(ext_d_toweek.shape)#torch.Size([16, 1, 7])
        # print(ext_d_tohour.shape)#torch.Size([16, 1, 24])
        # print(ext_r_toweek.shape)#torch.Size([16, 3, 7])
        # print(ext_r_tohour.shape)#torch.Size([16, 3, 24])
        x_w = self.bn(x_w)#torch.Size([16, 80, 80, 2])
        x_d = self.bn(x_d)#torch.Size([16, 80, 80, 6])
        x_r = self.bn(x_r)#torch.Size([16, 80, 80, 6])
        # print('xxx')
        # print(x_w.shape)
        # print(x_d.shape)
        # print(x_r.shape)
        x = torch.cat((x_w, x_d, x_r), -1)
        # print(x.shape)#torch.Size([16, 69, 69, 6])
        ext_toweek=torch.cat((ext_w_toweek,ext_d_toweek,ext_r_toweek),1)#torch.Size([16, 6, 7])
        ext_tohour=torch.cat((ext_w_tohour,ext_d_tohour,ext_r_tohour),1)#torch.Size([16, 6, 108])

        # print('890-')
        # print(ext_toweek.shape)
        # print(ext_tohour.shape)


        y = self.submodule(x,ext_toweek, ext_tohour)#torch.Size([16, 256, 80, 14])
        # print(y.shape)#torch.Size([16, 256, 69, 6])
        y1 = y[:, :, :, 0:1]  # torch.Size([16, 64, 80, 2])
        y2 = y[:, :, :, 1:2]  # torch.Size([16, 64, 80, 6])
        y3 = y[:, :, :, 2:3]  # torch.Size([16, 64, 80, 6])
        y4 = y[:, :, :, 3:6]  # torch.Size([16, 64, 80, 6])


        # print('看看y1,y2,y3的形状')
        # print(y1.shape)#torch.Size([16, 256, 69, 2])
        # print(y2.shape)#torch.Size([16, 256, 69, 1])
        # print(y3.shape)#torch.Size([16, 256, 69, 3])
        # print(y4.shape)  # torch.Size([16, 256, 69, 3])
        y1 = self.conv1(y1)  #torch.Size([16, 69, 69, 1])
        y2 = self.conv2(y2)  #torch.Size([16, 69, 69, 1])
        y3 = self.conv3(y3)  #torch.Size([16, 69, 69, 1])
        y4 = self.conv4(y4)  # torch.Size([16, 69, 69, 1])
        # print('conv')
        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)
        # 融合y1,y2,y3

        # print(y.shape)
        return y1+y2+y3+y4  #torch.Size([16, 80, 80, 1])



