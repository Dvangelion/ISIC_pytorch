import glob
import torch
import torch.nn as nn
from nets import senet
from nets import pnasnet
from configs import hyperparameters

#from nets.senet import se_resnext101_32x4d

network_fn_map = {
    'senet154': senet.senet154,
    'se_resnext101': senet.se_resnext101_32x4d
    'pnasnet': pnasnet.pnasnet5large
}



def get_network_function(network_name):
    #get network function and whether resume from exisiting checkpoint
    ckpt_list = glob.glob('./models/*.tar')
    resume = False

    if len(ckpt_list) > 0:
        resume = True
        ckpt_list.sort(key=lambda x:int(x[-11:-9]))
        print('Found checkpoint file in model dir, resumimg training from: %s' % ckpt_list[-1])
        network_fn = network_fn_map[network_name](num_classes=8, pretrained=None)
        return network_fn, resume
    
    else:
        print('Initializing model from imagenet')
        network_fn = network_fn_map[network_name](num_classes=1000, pretrained='imagenet')
        network_fn.last_linear = nn.Linear(network_fn.last_linear.in_features, hyperparameters.num_classes)
        return network_fn, resume
    