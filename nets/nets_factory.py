from nets import senet

#from nets.senet import se_resnext101_32x4d

network_fn_map = {
    'senet154': senet.senet154,
    'se_resnext101': senet.se_resnext101_32x4d
}



def get_network_function(network_name):
    

    return network_fn_map[network_name](num_classes=1000, pretrained='imagenet')
    