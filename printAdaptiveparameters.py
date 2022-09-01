import os
from Code.lib.model_RFNet import RFNet
import torch.backends.cudnn as cudnn
from Code.utils.options import opt
import torch



#set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU {}'.format(opt.gpu_id))

  
cudnn.benchmark = True


#build the model
model = RAFNet(32,50)
model.load_state_dict(torch.load('./Checkpoint/RFNet/RFNet_epoch_best.pth'))
print('---------------------------------------')
print('-------RAFNet DCF CA SA adapt Depth qaulity -----------')


with open('Alphas.txt', 'a') as file:
    file.write('---------------------------------------\n')
    file.write('-------case-2-----------')
    fusionNames = ['fu_0','fu_1','fu_2','fu_3','fu_4']
    for name,param in model.named_parameters():
        # print(name)
        for fusionName in fusionNames:
            alphaName = fusionName+'.alpha_C'
            if alphaName == name:
                percentage = param.detach().numpy()[0]
                print('Channel Contributes {}  in {}.'.format(percentage,fusionName))
                file.write('Channel Contributes {}  in {}.\n'.format(percentage,fusionName))

            alphaName = fusionName+'.alpha_S'
            if alphaName == name:
                percentage = param.detach().numpy()[0]
                print('Spatial Contributes {}  in {}.'.format(percentage,fusionName))
                file.write('Spatial Contributes {}  in {}.\n'.format(percentage,fusionName))

    file.write('---------------------------------------\n')
    file.write('\n')
    file.write('\n')
print('---------------------------------------')
