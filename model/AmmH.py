import torch.nn as nn
import torch
import gc 

from .resnet import ResNet,BasicBlock
from .embed import FeatureEmbed
from .transformer import IntraTransformer,InterTransformer
from .clshead import ClsHead
from .weight import vitWeight

class WMutilModel(nn.Module):
    def __init__(self,class_nums=2):
        super(WMutilModel,self).__init__()
        self.cnn_us = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2, include_top=False)
        self.cnn_th = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2, include_top=False)
        self.featemb_us = FeatureEmbed(512,196)
        self.featemb_th = FeatureEmbed(512,196)
        self.intratrans_us = IntraTransformer()
        self.intratrans_th = IntraTransformer()
        self.intertrans = InterTransformer(embed_dim=576)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.weight = vitWeight(in_features=392)
        self.head=ClsHead(196,class_nums)
    
    def forward(self,x_us,x_th):
        
        x_us = self.cnn_us(x_us)
        x_th = self.cnn_th(x_th)
      

        x_us = self.featemb_us(x_us)
        x_th = self.featemb_th(x_th)
      
        x_us = self.intratrans_us(x_us)
        x_th = self.intratrans_th(x_th)
        
        
        x_us_w = self.avgpool(x_us)
        x_th_w = self.avgpool(x_th)
      

        x_us_w = torch.flatten(x_us_w,1)
        x_th_w = torch.flatten(x_th_w,1)
    
        
        x_w = torch.cat((x_us_w,x_th_w),dim=-1)
        w_us,w_th = self.weight(x_w)

        w_us = torch.unsqueeze(w_us, dim=-1)
        w_th = torch.unsqueeze(w_th, dim=-1)

     
       
        x = w_us * x_us_w + w_th * x_th_w

        x = self.head(x)
      
        return x

