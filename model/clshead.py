import torch
import torch.nn as nn


class ClsHead(nn.Module):
    def __init__(self,embed_dim,num_classes=2):
        super(ClsHead,self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.drop = nn.Dropout(p=0.3)
    
    def forward(self,x):
        x = self.head(x)
        x = self.drop(x)
        return x