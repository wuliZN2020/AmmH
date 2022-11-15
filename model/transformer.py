from functools import partial
from collections import OrderedDict
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F


# Q: 一种正则化策略，暂时不用管
def drop_path(x, drop_prob=0., training=False):
    if drop_path == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    # 这里源代码中drop_prob=None
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self,x):
        return drop_path(x, self.drop_prob, self.training)

# 多头注意力机制的实现
class Attention(nn.Module):
    # 这里dim表示输入token的维度
    def __init__(self,dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop_ratio=0.,
            proj_drop_ratio=0.):

            super(Attention, self).__init__()

            self.num_heads = num_heads
            head_dim = dim // num_heads
            # Q: 这里为什么要用or?
            # A: 如果qk_scale=None,则self.scale=head_dim ** -0.5;否则self.scale=qk_scale
            self.scale = qk_scale or head_dim ** -0.5 
            self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop_ratio)

            # Q: proj做什么操作？
            # A：
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop_ratio)
    
    def forward(self,x):

        B, N, C = x.shape
        
        # [B, N, C] -> [B, N, 3*C]
        qkv = self.qkv(x)
        # [B, N, 3*C] -> [B, N, 3, num_heads, C//num_heads]
        # qkv = qkv.reshape(B,N,3,self.num_heads,C//self.num_heads)
        qkv = qkv.reshape(B,N,3,self.num_heads,torch.div(C,self.num_heads,rounding_mode='trunc'))
        # [B, N, 3, num_heads, C//num_heads] -> [3, B, num_heads, N, C//num_heads]
        qkv = qkv.permute(2,0,3,1,4)
        # q.shape = [B, num_heads, N, C//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attn.shape = [B, num_heads, N, N] 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x.shape = [B, num_heads, N, C//num_heads] 
        x = attn @ v
        # [B, num_heads, N, C//num_heads] -> [B, N, num_heads, C//num_heads]
        x = x.permute(0,2,1,3)
        # [B, N, num_heads, C//num_heads] -> [B, N, C] 该步骤相当于将多头注意力机制最后得到的concact
        x = x.reshape(B, N, C)
        x = self.proj(x) # 全连接层
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    def __init__(self,
                in_features, 
                hidden_features=None, 
                out_features=None, 
                act_layer=nn.GELU, 
                drop=0.):
        super(Mlp,self).__init__()
        # 如果out_features=None,则out_features=out_features;否则out_features=in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4., # 在MLP层会先将全连接层展成原来的4倍
        qkv_bias=False,
        qk_scale=None,
        drop_ratio=0.,
        attn_drop_ratio=0.,
        drop_path_ratio=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # nn.Identity() 输入=输出，不做任何的改变。
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 =norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    
    def forward(self, x):
        # 源代码
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # return x
        
        out1 = self.norm1(x)
        out1 = self.attn(out1)
        out1 = self.drop_path(out1)
        out1 = x + out1

        out2 = self.norm2(out1)
        out2 = self.mlp(out2)
        out2 = self.drop_path(out2)
        out = out1 + out2

        return out

#  [B, N ,C] -> [B, N ,C]
class IntraTransformer(nn.Module):
    # depth表示在Transformer中重复堆叠Block的次数
    # representation_size表示在最后用于分类的MLP Head中是否使用pre-logits层
    # num_features表示FeatureEmbed中特征图的数目,该参数主要是用于添加位置信息

    def __init__(self,num_features=196, embed_dim=576,depth=4,num_heads=1,mlp_ratio=4.0,
                qkv_bias=True, qk_scale=None, drop_ratio=0.,
                attn_drop_ratio=0., drop_path_ratio=0., 
                norm_layer=None, act_layer=None):
                super(IntraTransformer,self).__init__()

                self.num_features = num_features
                self.embed_dim = embed_dim
                # partial函数目的为传入默认参数
                norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
                act_layer = act_layer or nn.GELU

                
                self.pos_embed = nn.Parameter(torch.zeros(1, num_features , embed_dim))
                self.pos_drop = nn.Dropout(p=drop_ratio)

                # dpr 为一个递增的等差序列，在Transformer的每一个Block中，其drop_path_ratio是不同的
                dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
                self.blocks = nn.Sequential(*[
                    EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                    norm_layer=norm_layer,act_layer=act_layer) for i in range(depth)
                ])

                self.norm = norm_layer(embed_dim)
                # Weight init
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
                
                


    def forward(self, x):
        # 添加位置信息
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        
        return x

# [B, N , 2 * C] -> [B, 2]
class InterTransformer(nn.Module):
    def __init__(self,num_classes=2,num_features=196, embed_dim=576*2,depth=4,num_heads=8,mlp_ratio=4.0,
                qkv_bias=True, qk_scale=None, drop_ratio=0.,
                attn_drop_ratio=0., drop_path_ratio=0., 
                norm_layer=None, act_layer=None):

                super(InterTransformer,self).__init__()
                self.num_classes = num_classes
                self.embed_dim = embed_dim
                self.num_tokens = 1
                norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
                act_layer = act_layer or nn.GELU

                num_features = num_features

                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_features + self.num_tokens, embed_dim))
                self.pos_drop = nn.Dropout(p=drop_ratio)

                dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
                self.blocks = nn.Sequential(*[
                    EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                            norm_layer=norm_layer, act_layer=act_layer)
                    for i in range(depth)
                ])
                self.norm = norm_layer(embed_dim)
                
                # Weight init
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
                nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self,x):
        # # [1, 1, C] -> [B, 1, C]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        #  [B, N, C] -> [B, N+1, C]
        x = torch.cat((cls_token, x), dim=1) 
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:,0] # 取出第一个向量用于分类

        return x




if __name__ == "__main__":
    pass



