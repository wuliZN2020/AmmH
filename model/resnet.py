"""
一共分为三个部分：
1、ResNet18作为BackBone：提取局部特征
2、IntraTransformer作为模态内编码器：提取模态内全局特征
3、InterTransformer作为模态间编码器：提取模态间全局特征
IDEA： TODO：可以尝试在Transformer之后再加上注意力机制对不同模态赋予不同权重

## TODO:所有的基础模块中都没有进行权重初始化
"""
from xml.etree.ElementInclude import include
import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    # Q：此处的参数expansion表示什么？
    # A：表示每一个Block中卷积层的卷积核个数是否发生变化，在ResNet18，34中并没有发生变化，在更深层次的网络发生了变化
    expansion = 1
    # Q：downsample表示什么？
    # A：该参数的设置是为了表示保证输入输出一致，在论文图中虚线部分所表示的捷径残差层。

    # Q：BatchNorm2d中传入的参数是什么？
    # A：传入参数为输入特征矩阵的通道数。
    def __init__(self,in_channel,out_channel,stride=1,downsample=None,**kwargs):
        super(BasicBlock,self).__init__()
        # [1, 3, 224, 224] -> [1, 3, 224, 224]

        # 在使用BN层时要将nn.Conv2d的bias设置为Flase
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        

        self.downsample = downsample
    
    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)

        return out

class BottleNeck(nn.Module):
    # 在resnet50以上的网络结构中，每一个卷积小组由3个卷积层构成，conv1，conv2通道数不改变，conv3通道数目变成原来的四倍
    expansion = 4 

    # Q: groups=1,width_per_group=64这两个参数有什么含义？
    # A：groups=1,width_per_group=64这是在RestNeXt中才会用到的参数
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(BottleNeck,self).__init__()

        # squeeze channels:压缩通道数，不改变图像大小
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        #self.conv2中的stride是根据传入的参数决定的，虚线残差结构和实线残差结构不一样
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                                kernel_size=3,stride=stride,bias=False,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        # unsqueeze channels:解压通道数，不改变图像大小
        self.conv3 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel*self.expansion,
                                kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        #Q: inplace =True有什么含义？
        #A: inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出.
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 注意第3层结束之后没有relu函数
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return(out)

class ResNet(nn.Module):
    # Q：参数block_nums的含义是什么？
    # A：定义ResNet18，ResNet34，使用BasicBlock；定义ResNet50，ResNet101使用BottleNeck

    # Q：参数block_nums的含义是什么？
    # A：参数block_nums是一个列表结构，为原论文中的“X”后对应的数目

    # include_top是为了在ResNet基础上搭建更加复杂的网络结构
    def __init__(self,block,block_nums,num_classes,include_top=False):
        super(ResNet,self).__init__()
        self.include_top = include_top
        self.in_channel = 64 # 在ResNet中通过max pool之后得到的网络的通道数目，所有ResNet都为64.

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.in_channel,stride=2,kernel_size=3,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # Q:MaxPool2d的输出图的尺寸计算？
        # A：
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        # model(1) > layer(4) > block([3,4,6,3]) > conv(2,3)
        # Q：这里的参数64表示什么？
        # A：64表示的是每一个block中的通道数，举个例子：在layer1中有3个BasicBlock，每个BasicBlock中的2个conv通道数是保持一致的
        #    但是在BottleNeck中，每个Block中会存在3个conv，conv3的通道数目变为原来的4倍
        self.layer1 = self._make_layer(block,64,block_nums[0]) # 在第一层中不做下采样操作，所以stride=1
        self.layer2 = self._make_layer(block,128,block_nums[1],stride=2)
        self.layer3 = self._make_layer(block,256,block_nums[2],stride=2)
        self.layer4 = self._make_layer(block,512,block_nums[3],stride=2)

        # Q: nn.AdaptiveAvgPool2d操作会得到什么结果？
        # A：
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # output size = (1,1)
            self.fc = nn.Linear(512*block.expansion, num_classes)
            self.drop = nn.Dropout(p=0.5)

    def _make_layer(self,block,channel,block_num,stride=1):
        downsample = None
        # 对于layer1默认是为stride=1，但是layer2,3,4中stride=2，为了保持输出与X可以进行+运算，需要对X作downsample操作
        # Q:这里有点绕？没太理解。
        # A：
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel, channel,downsample=downsample,stride=stride))
        self.in_channel = channel * block.expansion

        for i in range(1,block_num):
            layers.append(block(self.in_channel, channel))
        
        # (*)中的*是干什么的？
        return nn.Sequential(*layers)
    
    def forward(self,x):

       
        # [1, 3, 224, 224] -> [1, 64, 114, 114]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
 
        # [1, 64, 114, 114] -> [1, 64, 57, 57]
        x = self.maxpool(x)
    
        # [1, 64, 57, 57] -> [1, 64, 57, 57] 
        x = self.layer1(x)

        # [1, 64, 57, 57] -> [1, 128, 29, 29]
        x = self.layer2(x)

        # [1, 128, 29, 29] -> [1, 256, 15, 15] 
        x = self.layer3(x)

        # [1, 256, 15, 15] -> [1, 512, 8, 8]
        x = self.layer4(x)

        if self.include_top:
        
            # [1, 512, 8, 8] -> [1, 512, 1, 1]
            x = self.avgpool(x)
            # [1, 512, 1, 1] -> [1, 512]
            x = torch.flatten(x,1)
            # [1, 512]-> [1, 2]
            x =self.fc(x)
            
        return x
            
         

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2, include_top=True)






            












