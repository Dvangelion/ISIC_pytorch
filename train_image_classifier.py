import os
import glob
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.preprocessing import preprocess_fn
from dataset.ISIC import ISICDataLoader, get_filenames_and_labels, get_validation_filenames_and_labels
from nets import nets_factory
from configs import hyperparameters




network_fn, resume = nets_factory.get_network_function('senet154')
epoch = 1

#load exisiting checkpoint
if resume:
    ckpt_list = glob.glob('./models/*.tar')
    ckpt_list.sort(key=lambda x:int(x[-11:-9]))
    checkpoint = torch.load(ckpt_list[-1])
    network_fn.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    #acc_history = checkpoint['acc_history']


network_fn = network_fn.cuda()

#network_fn need to allocate to gpu before defining optimizer
optimizer = optim.Adam(network_fn.parameters(), lr=hyperparameters.initial_learning_rate)
if resume:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0000, 0.3512, 1.3608, 5.2157, 
                                1.7233, 18.9205, 17.8735, 7.2006],device='cuda'))

#try cyclic LR in PNASnet?
scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters.num_epochs_per_decay, gamma=hyperparameters.learning_rate_decay)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,75,100], gamma=0.2)
#scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0., max_lr=0.01,step_size_up=797)

photo_filenames, photo_labels, class_names = get_filenames_and_labels()
validation_filenames, validation_labels = get_validation_filenames_and_labels(hyperparameters.validation_per_class)

preprocessing_fn = preprocess_fn.senet_preprocessing(is_training=True)
val_preprocessing_fn = preprocess_fn.senet_preprocessing(is_training=False)

train_dataloader = ISICDataLoader(photo_filenames, photo_labels, transform=preprocessing_fn, process='train')
train_loader = DataLoader(train_dataloader, batch_size=hyperparameters.batch_size, 
                                num_workers=hyperparameters.num_workers, pin_memory=True, shuffle=True)
                                 

validation_dataloader = ISICDataLoader(validation_filenames, validation_labels, transform=val_preprocessing_fn, 
                                process='validation')
validation_loader = DataLoader(validation_dataloader, batch_size=hyperparameters.batch_size, 
                                    num_workers=hyperparameters.num_workers, pin_memory=True, shuffle=True)

def save_model(state):
    #save model to model_dir
    model_dir = './models/'
    ckpt_name = 'model.ckpt-%.3d%s' % (state['epoch'], '.pth.tar')
    ckpt_dir = os.path.join(model_dir, ckpt_name)
    torch.save(state, ckpt_dir)
    ckpt_list = glob.glob('./models/*.tar')
    
    if len(ckpt_list) > 5:
        ckpt_list.sort(key=lambda x:int(x[-11:-9]))
        os.remove(ckpt_list[0])
    
def eval_model(model):

    #model.eval()
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
    
    print('Weighted Mean ACC: %.3f %%' % mean_wacc)

    return acc_per_class
    

#training epoch
for epoch in range(epoch, hyperparameters.max_epochs+1):
    running_loss = 0.0
    acc_history = []
    for batch_i, (data, label) in tqdm(enumerate(train_loader), total = len(train_loader)):
        data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()
        output = network_fn(data)
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()
        #cyclical_LR
        #scheduler.step()
        running_loss += loss.item()

        #calculate batch loss per 528 batchs, 3 times per epoch
        if batch_i % 528 == 527:
        #if batch_i % 211 == 210:
            print('epoch %d, batch_num %3d loss: %.3f' %
            (epoch, batch_i + 1, running_loss / 528))
            running_loss = 0.0

            for param_group in optimizer.param_groups:
                print('current learning rate: ', param_group['lr'])
    
    #step_LR        
    scheduler.step()

    if (epoch-1) % hyperparameters.num_epochs_per_eval == 0 and epoch > 1:
        acc_history.append(eval_model(network_fn))
        
    
    if (epoch-1) % hyperparameters.save_epochs == 0 and epoch > 1:
        save_model(
            {'epoch':epoch,
            'model_state_dict':network_fn.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss,
            'acc_history':acc_history
            }
        )
    #print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}')

#TODO: confusion matrix, add pnasnet model, load checkpoint if exisits, test_data inference

print('Optmization Done.')