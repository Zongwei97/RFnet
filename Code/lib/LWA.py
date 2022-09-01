import torch
import torch.nn as nn
import math
from typing import Optional
from torch import Tensor
import torch.nn.functional as F

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class lwa(nn.Module):
    def __init__(self,channel,outsize):
        super().__init__()

        #self.rgbd = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
        self.dept = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
        self.rgb  = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)

        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(_ConvBNReLU(channel, 24, 1, 1),_ConvBNSig(24,outsize,1,1))

    
    def forward(self,rgb,dep):

        assert rgb.size() == dep.size()

        rgbd = rgb+dep
        m_batchsize,C,width ,height = rgb.size()

        proj_rgb  = self.rgb(rgb).view(m_batchsize,-1,height*width).permute(0,2,1) # B X (H*W) X C
        proj_dep  = self.dept(dep).view(m_batchsize,-1,height*width) # B X C x (H*W)
        energy    = torch.bmm(proj_rgb,proj_dep)/math.sqrt(C)  #B X (H*W) X (H*W)
        attention1 = self.softmax1(energy) #B X (H*W) X (H*W) 


        att_r = torch.bmm(proj_rgb.permute(0,2,1),attention1 )
        att_b = torch.bmm(proj_dep,attention1 )
        #proj_rgbd = self.rgbd(rgbd).view(m_batchsize,-1,height*width) # B X C X (H*W) 
        #attention2 = torch.bmm(proj_rgbd,attention1.permute(0,2,1) )
        attention2 = att_r + att_b
        output = attention2.view(m_batchsize,C,width,height) + rgbd

        GapOut = self.GAP(output)
        gate = self.mlp(GapOut)

        return gate




if __name__ == "__main__":
    K = lwa(20,5)
    K.to('cuda')
    rgb   = torch.rand([2,64,64,64]).cuda()
    depth = torch.rand((2,64,64,64)).cuda()
    model = lwa(64,64).cuda()
    k = model(rgb,depth)
    print(k.size())

