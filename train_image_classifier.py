import os
import glob
import pickle
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.preprocessing import preprocess_fn
from dataset.ISIC import ISICDataLoader, get_filenames_and_labels, get_validation_filenames_and_labels
from nets import nets_factory
from configs import hyperparameters




network_fn = nets_factory.get_network_function('senet154')
network_fn.last_linear = nn.Linear(network_fn.last_linear.in_features, hyperparameters.num_classes)

network_fn = network_fn.cuda()

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0000, 0.3512, 1.3608, 5.2157, 
                                    1.7233, 18.9205, 17.8735, 7.2006],device='cuda'))

optimizer = optim.Adam(network_fn.parameters(), lr=hyperparameters.initial_learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters.num_epochs_per_decay, gamma=hyperparameters.gamma)

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
                                    num_workers=hyperparameters.num_workers, pin_memory=True, shuffle=False)

def save_model(state):
    model_dir = './models/'
    ckpt_name = 'model.ckpt-%.3d%s' % (state['epoch'], '.pth.tar')
    ckpt_dir = os.path.join(model_dir, ckpt_name)
    torch.save(state, ckpt_dir)
    ckpt_list = glob.glob('./models/*.tar')
    
    if len(ckpt_list) > 5:
        ckpt_list.sort(key=lambda x:int(x[-11:-9]))
        os.remove(ckpt_list[0])

def eval_model():
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    acc_per_class = list(0. for i in range(hyperparameters.num_classes))

    with torch.no_grad():
        for (data, labels) in validation_loader:
            data, labels = data.cuda(), labels.cuda()
            outputs = network_fn(data)
            predicted = torch.argmax(outputs.data, dim=1)
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

    return acc_per_class
    


for epoch in range(1, hyperparameters.max_epochs+1):
    running_loss = 0.0
    #epoch_loss = []
    acc_history = []
    for batch_i, (data, label) in tqdm(enumerate(train_loader), total = len(train_loader)):
        data, label = data.cuda(), label.cuda()

        optimizer.zero_grad()
        output = network_fn(data)
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #epoch_loss.append

        #eval per 528 batchs, 3 times per epoch
        if batch_i % 528 == 527:
            acc_history.append(eval_model())
            print('epoch %d, batch_num %3d loss: %.3f' %
            (epoch, batch_i + 1, running_loss / 528))
            running_loss = 0.0
    
    scheduler.step()

    if (epoch-1) % hyperparameters.num_epochs_per_eval == 0 and epoch > 1:
        eval_model()
    
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

#TODO: recall metric (per class), save model and acc metric, add pnasnet model, save acc_history_file when saving model

acc_history_file = open('./models/acc_history.txt', 'wb')
pickle.dump(acc_history, acc_history_file)

print('Optmization Done.')