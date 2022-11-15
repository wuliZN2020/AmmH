import torch
import torch.nn as nn

# TODO [B, 512, 8, 8] -> [B, N ,C]
class FeatureEmbed(nn.Module):
    # 这里初始化的参数in_channel为特征图通道的个数,feature_nums为输出的特征通道数目,ratio表示对输入进行上采样的倍数
    def __init__(self,in_channel,feature_nums,feature_size=[8,8],ratio=3):
 
        super(FeatureEmbed,self).__init__()
        self.feature_nums = feature_nums
        self.feature_size = feature_size # embed_dim=self.feature_size[0] * self.feature_size[0] * ratio * ratio
        self.embed_dim = feature_size[0] * ratio * feature_size[1] * ratio
        self.upconv = nn.ConvTranspose2d(in_channels=in_channel,out_channels=feature_nums,kernel_size=ratio,stride=ratio)
    def forward(self,x):
        x = self.upconv(x) # [B, in_channel, H, W] -> [B, feature_nums, ratio/2*H, ratio/2*W] 
        x = torch.flatten(x,2) # [B, feature_nums, ratio*H, ratio*W] -> [B, feature_nums,ratio*H *ratio*W]= [B, feature_nums,embed_dim]
        return x