import torch
import torch.nn as nn


#实现Vit的权重
class vitWeight(nn.Module):
    def __init__(self,in_features,num_models=2):
        super(vitWeight,self).__init__()
        self.fc1 = nn.Linear(in_features,torch.div(in_features,2,rounding_mode='trunc'))
        self.fc2 = nn.Linear(torch.div(in_features,2,rounding_mode='trunc'),torch.div(in_features,4,rounding_mode='trunc'))
        self.fc3 = nn.Linear(torch.div(in_features,4,rounding_mode='trunc'),num_models)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x[:,0],x[:,1]
