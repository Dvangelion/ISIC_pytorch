import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.preprocessing import preprocess_fn
from dataset.ISIC import ISICDataLoader, get_filenames_and_labels, get_validation_filenames_and_labels
from configs import hyperparameters
from nets import nets_factory
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import confusion_matrix

gpu_devices = '0,1,2,3,4,5,6,7'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

def save_model(state):
    #save model to model_dir
    model_dir = './models/'
    ckpt_name = 'model.ckpt-%.3d%s' % (state['epoch'], '.pth.tar')
    ckpt_dir = os.path.join(model_dir, ckpt_name)
    torch.save(state, ckpt_dir)
    ckpt_list = glob.glob('./models/*.tar')
    
    if len(ckpt_list) > 5:
        ckpt_list.sort(key=lambda x:int(x[-11:-8]))
        os.remove(ckpt_list[0])

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    quarter_batch = len(train_loader) // 4
    

    for batch_i, (data, label) in tqdm(enumerate(train_loader), total = len(train_loader)):
        data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        #calculate batch loss per 1/4 batchs, 4 times per epoch
        if batch_i % quarter_batch == (quarter_batch-1):
            print('epoch %d, batch_num %3d loss: %.3f' %
            (epoch, batch_i + 1, running_loss / quarter_batch))
            running_loss = 0.0

            for param_group in optimizer.param_groups:
                print('current learning rate: ', param_group['lr'])

    save_model({'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss
                })

    return loss

# def adjust_learning_rate(initial_lr, optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = initial_lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def validate(validation_loader, model, class_names, best_mean_wacc):
    model.eval()
    #evaluates accuracy per class of model
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    acc_per_class = list(0. for i in range(hyperparameters.num_classes))
    prediction_list = []
    label_list = []

    with torch.no_grad():
        for (data, labels) in validation_loader:
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            predicted = torch.argmax(outputs.data, dim=1)

            prediction_list.extend(predicted.cpu().tolist())
            label_list.extend(labels.cpu().tolist())
            #reduced_label = torch.argmax(label, dim=1)
            correct = (predicted == labels)
            for i in range(labels.size(0)):
                #label = reduced_label[i]
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    for i in range(len(class_names)):
        try:
            class_acc = 100* class_correct[i]/class_total[i] 
            acc_per_class[i] = class_acc
            print('Accuracy of %4s : %.2f %%' % (
            class_names[i], class_acc))
        except:
            print('Accuracy of %4s : %.2f %%' % (
            class_names[i], 0.))

    mean_acc = np.mean(acc_per_class)
    print('Mean accuracy: %.3f %%' % mean_acc)

    conf = confusion_matrix(prediction_list, label_list)
    wacc = conf.diagonal()/conf.sum(axis=1)
    mean_wacc = np.mean(wacc)
    
    print('Weighted Mean ACC: %.3f' % mean_wacc)
    
    if mean_wacc > best_mean_wacc['mean_wacc']:
        print('Found better mean_wacc at epoch %d' % epoch)
        best_mean_wacc['mean_wacc'] = mean_wacc
        best_mean_wacc['epoch'] = epoch

    return best_mean_wacc



print("Initialize Model...")
# Construct Model
network_fn, resume = nets_factory.get_network_function('pnasnet')
epoch = 1

#load exisiting checkpoint
if resume:
    ckpt_list = glob.glob('./models/*.tar')
    ckpt_list.sort(key=lambda x:int(x[-11:-8]))
    checkpoint = torch.load(ckpt_list[-1])
    state_dict = checkpoint['model_state_dict']
    #Dataparallel changed names in state dict, map it back
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    network_fn.load_state_dict(new_state_dict)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    #acc_history = checkpoint['acc_history']


network_fn = network_fn.cuda()
network_fn = nn.DataParallel(network_fn)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0000, 0.3512, 1.3608, 5.2157, 
                                        1.7233, 18.9205, 17.8735, 7.2006],device='cuda'))

#optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(network_fn.parameters(), lr=hyperparameters.initial_learning_rate)
if resume:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,75,100], gamma=0.2)

print("Initialize Dataloaders...")
# Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
photo_filenames, photo_labels, class_names = get_filenames_and_labels()
validation_filenames, validation_labels = get_validation_filenames_and_labels(hyperparameters.validation_per_class)

preprocessing_fn = preprocess_fn.senet_preprocessing(is_training=True)
val_preprocessing_fn = preprocess_fn.senet_preprocessing(is_training=False)

# Initialize Datasets. STL10 will automatically download if not present
trainset = ISICDataLoader(photo_filenames, photo_labels, transform=preprocessing_fn, process='train')
valset = ISICDataLoader(validation_filenames, validation_labels, transform=val_preprocessing_fn, process='validation')

# Create DistributedSampler to handle distributing the dataset across nodes when training
# This can only be called after torch.distributed.init_process_group is called
#train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

# Create the Dataloaders to feed data to the training and validation steps
train_loader = torch.utils.data.DataLoader(trainset, batch_size=hyperparameters.batch_size, shuffle=True, num_workers=hyperparameters.num_workers, pin_memory=False)
val_loader = torch.utils.data.DataLoader(valset, batch_size=hyperparameters.batch_size, shuffle=False, num_workers=hyperparameters.num_workers, pin_memory=False)

best_mean_wacc = {'epoch':0, 'mean_wacc':0}

for epoch in range(epoch, hyperparameters.max_epochs+1):

    # train for one epoch
    print("\nBegin Training @ Epoch {}".format(epoch))
    loss = train(train_loader, network_fn, criterion, optimizer, epoch)

    # evaluate on validation set
    print("Begin Validation @ Epoch {}".format(epoch))
    best_mean_wacc = validate(val_loader, network_fn, class_names, best_mean_wacc)

    # remember best prec@1 and save checkpoint if desired
    # is_best = prec1 > best_prec1
    #best_prec1 = max(prec1, best_prec1)

    scheduler.step()
    
    

    #print("Epoch Summary: ")
    #print("\tEpoch Accuracy: {}".format(prec1))
    #print("\tBest Accuracy: {}".format(best_prec1))
    print('Best Mean WACC: {:.3f}'.format(best_mean_wacc['mean_wacc']), 
            'Epoch: {}'.format(best_mean_wacc['epoch']))
