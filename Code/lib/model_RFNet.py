import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from Code.lib.res2net_v1b_base import Res2Net_model
from Code.lib.LWA import lwa

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool




class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



#Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x




def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


class TSA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1,dilation=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid())
        
        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=3,dilation=3),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.spatial_attention3 = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=5,dilation=5),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.sp_conv = nn.Conv2d(3, 1, 1, padding=0)

    def forward(self,x_r):
        
        SA1 = self.spatial_attention1(x_r)
        SA2 = self.spatial_attention2(x_r)
        SA3 = self.spatial_attention3(x_r)

        SA_final  = self.sp_conv(torch.cat([SA1,SA2,SA3],dim=1))

        return SA_final

class AF0(nn.Module):
    def __init__(self,indim,outdim):
        super(AF0, self).__init__()


        self.alpha_C = nn.parameter.Parameter(torch.Tensor(1),requires_grad=True)
        self.alpha_S = nn.parameter.Parameter(torch.Tensor(1),requires_grad=True)

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(indim, indim, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())


        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(indim, indim, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())
        


        self.spatial_attention_rgb = TSA()
        self.spatial_attention_depth = TSA()


        self.reset_parameters()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def reset_parameters(self):
        ones(self.alpha_C)
        ones(self.alpha_S)

    def forward(self, x3_r,x3_d,Quality_D):

        #########Channel Part like DCF  ##############

        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        SCA_3_o_ca = x3_r * SCA_ca.expand_as(x3_r)

        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        SCA_3d_o_ca = x3_d * SCA_d_ca.expand_as(x3_d)

        Co_ca3_ca = torch.softmax(SCA_ca + SCA_d_ca,dim=1)

        SCA_3_co_ca = x3_r * Co_ca3_ca.expand_as(x3_r)
        SCA_3d_co_ca= x3_d * Co_ca3_ca.expand_as(x3_d)

        CR_fea3_rgb_ca = SCA_3_o_ca + SCA_3_co_ca
        CR_fea3_d_ca = SCA_3d_o_ca + SCA_3d_co_ca


        ######### Spatial Part  ##############

        avg_out_r = torch.mean(x3_r, dim=1, keepdim=True)
        max_out_r, _ = torch.max(x3_r, dim=1, keepdim=True)
        x_r = torch.cat([avg_out_r, max_out_r], dim=1)

        SCA_sa = self.spatial_attention_rgb(x_r)
        SCA_3_o_sa = x3_r * SCA_sa


        avg_out_d = torch.mean(x3_d, dim=1, keepdim=True)
        max_out_d, _ = torch.max(x3_d, dim=1, keepdim=True)
        x_d = torch.cat([avg_out_d, max_out_d], dim=1)
        
        SCA_d_sa = self.spatial_attention_depth(x_d)
        SCA_3d_o_sa = x3_d * SCA_d_sa

        Co_ca3_sa = torch.softmax(SCA_sa + SCA_d_sa,dim=1)

        SCA_3_co_sa = x3_r * Co_ca3_sa
        SCA_3d_co_sa= x3_d * Co_ca3_sa

        CR_fea3_rgb_sa = SCA_3_o_sa + SCA_3_co_sa
        CR_fea3_d_sa = SCA_3d_o_sa + SCA_3d_co_sa

        CR_fea3_rgb = self.alpha_C * CR_fea3_rgb_ca + self.alpha_S * CR_fea3_rgb_sa
        CR_fea3_d   = self.alpha_C * CR_fea3_d_ca   + self.alpha_S * CR_fea3_d_sa

        CR_fea3 = CR_fea3_rgb + (Quality_D * CR_fea3_d)

        return CR_fea3

class AF(nn.Module):
    def __init__(self,indim,outdim):
        super(AF, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)

        self.alpha_S = nn.parameter.Parameter(torch.Tensor(1),requires_grad=True)
        self.alpha_C = nn.parameter.Parameter(torch.Tensor(1),requires_grad=True)

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(indim, indim, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())


        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(indim, indim, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())
        


        self.spatial_attention_rgb = TSA()
        self.spatial_attention_depth = TSA()



        self.reset_parameters()
        self.layer_ful2 = nn.Sequential(nn.Conv2d(indim+outdim//2, outdim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(outdim),act_fn,)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def reset_parameters(self):
        ones(self.alpha_C)
        ones(self.alpha_S)

    def forward(self, x3_r,x3_d,Quality_D,xx):


        ######### Channel Part like DCF  ##############
        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        SCA_3_o_ca = x3_r * SCA_ca.expand_as(x3_r)

        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        SCA_3d_o_ca = x3_d * SCA_d_ca.expand_as(x3_d)

        Co_ca3_ca = torch.softmax(SCA_ca + SCA_d_ca,dim=1)

        SCA_3_co_ca = x3_r * Co_ca3_ca.expand_as(x3_r)
        SCA_3d_co_ca= x3_d * Co_ca3_ca.expand_as(x3_d)

        CR_fea3_rgb_ca = SCA_3_o_ca + SCA_3_co_ca
        CR_fea3_d_ca = SCA_3d_o_ca + SCA_3d_co_ca


        ######### Spatial Part  ##############

        avg_out_r = torch.mean(x3_r, dim=1, keepdim=True)
        max_out_r, _ = torch.max(x3_r, dim=1, keepdim=True)
        x_r = torch.cat([avg_out_r, max_out_r], dim=1)

        SCA_sa = self.spatial_attention_rgb(x_r)
        SCA_3_o_sa = x3_r * SCA_sa


        avg_out_d = torch.mean(x3_d, dim=1, keepdim=True)
        max_out_d, _ = torch.max(x3_d, dim=1, keepdim=True)
        x_d = torch.cat([avg_out_d, max_out_d], dim=1)
        
        SCA_d_sa = self.spatial_attention_depth(x_d)
        SCA_3d_o_sa = x3_d * SCA_d_sa

        Co_ca3_sa = torch.softmax(SCA_sa + SCA_d_sa,dim=1)

        SCA_3_co_sa = x3_r * Co_ca3_sa
        SCA_3d_co_sa= x3_d * Co_ca3_sa

        CR_fea3_rgb_sa = SCA_3_o_sa + SCA_3_co_sa
        CR_fea3_d_sa = SCA_3d_o_sa + SCA_3d_co_sa


        CR_fea3_rgb = self.alpha_C * CR_fea3_rgb_ca + self.alpha_S * CR_fea3_rgb_sa
        CR_fea3_d   = self.alpha_C * CR_fea3_d_ca   + self.alpha_S * CR_fea3_d_sa

        CR_fea3 = CR_fea3_rgb +  (Quality_D * CR_fea3_d)
        out2 = self.layer_ful2(torch.cat([CR_fea3,xx],dim=1))

        return out2

    

  
   
###############################################################################

class RFNet(nn.Module):
    def __init__(self, channel=32,ind=50):
        super(RFNet, self).__init__()
        
       
        self.relu = nn.ReLU(inplace=True)
        
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        #Backbone model
        self.layer_rgb  = Res2Net_model(ind)
        self.layer_dep  = Res2Net_model(ind)
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)
        
        ###############################################
        # Layer-Wise Attention#
        ###############################################
        self.depth_qaulity = lwa(256,5)


        ###############################################
        # funsion encoders #
        ###############################################

        self.fu_0 = AF0(64, 64)#
        
        self.fu_1 = AF(256, 128) #MixedFusion_Block_IMfusion
        self.pool_fu_1 = maxpool()
        
        self.fu_2 = AF(512, 256)
        self.pool_fu_2 = maxpool()
        
        self.fu_3 = AF(1024, 512)
        self.pool_fu_3 = maxpool()

        self.fu_4 = AF(2048, 1024)
        self.pool_fu_4 = maxpool()
        
        
        ###############################################
        # decoders #
        ###############################################       
        ## fusion
        self.ful_gcm_4    = GCM(1024,  channel)        
        self.ful_gcm_3    = GCM(512+channel,  channel)
        self.ful_gcm_2    = GCM(256+channel,  channel)
        self.ful_gcm_1    = GCM(128+channel,  channel)
        self.ful_gcm_0    = GCM(64+channel,  channel)        
        self.ful_conv_out = nn.Conv2d(channel, 1, 1)
        
        
                
    def forward(self, imgs, depths):
        
        img_0, img_1, img_2, img_3, img_4 = self.layer_rgb(imgs)
        dep_0, dep_1, dep_2, dep_3, dep_4 = self.layer_dep(self.layer_dep0(depths))


        lambdaLWA = self.depth_qaulity(img_1,dep_1)
        ####################################################
        ## fusion
        ####################################################
        ful_0    = self.fu_0(img_0, dep_0, lambdaLWA[:,0:1,...])
        ful_1    = self.fu_1(img_1, dep_1, lambdaLWA[:,1:2,...], ful_0)
        ful_2    = self.fu_2(img_2, dep_2, lambdaLWA[:,2:3,...], self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(img_3, dep_3, lambdaLWA[:,3:4,...], self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(img_4, dep_4, lambdaLWA[:,4:5,...], self.pool_fu_3(ful_3))

        ####################################################
        ## decoder fusion
        ####################################################        
        #
        x_ful_42    = self.ful_gcm_4(ful_4)
        
        x_ful_3_cat = torch.cat([ful_3, self.upsample_2(x_ful_42)], dim=1)
        x_ful_32    = self.ful_gcm_3(x_ful_3_cat)
        
        x_ful_2_cat = torch.cat([ful_2, self.upsample_2(x_ful_32)], dim=1)
        x_ful_22    = self.ful_gcm_2(x_ful_2_cat)        

        x_ful_1_cat = torch.cat([ful_1, self.upsample_2(x_ful_22)], dim=1)
        x_ful_12    = self.ful_gcm_1(x_ful_1_cat)     

        x_ful_0_cat = torch.cat([ful_0, x_ful_12], dim=1)
        x_ful_02    = self.ful_gcm_0(x_ful_0_cat)     
        ful_out     = self.upsample_4(self.ful_conv_out(x_ful_02))


        return ful_out
    
    

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    


# Test code 

if __name__ == '__main__':
    rgb = torch.rand((2,3,352,352)).cuda()
    depth = torch.rand((2,1,352,352)).cuda()
    model = RFNet(32,50).cuda()
    l = model(rgb,depth)
    print(l.size())