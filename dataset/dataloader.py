import os
import cv2
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image


_NUM_VALIDATION = 2553
_ISIC_DATA_FILE = './dataset/ISIC/ISIC_2019_Training_GroundTruth.csv'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6337, 0.6060, 0.5936],
                            std=[0.1393, 0.1832, 0.1970])
    ]),
    
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6337, 0.6060, 0.5936],
                            std=[0.1393, 0.1832, 0.1970])
    ]),

    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6337, 0.6060, 0.5936],
                            std=[0.1393, 0.1832, 0.1970])
    ])
}



class ISICDataset(Dataset):
    def __init__(self, data_file, process=None, transform=None):
        self.data_directory = './dataset/ISIC/Preprocessed_ISIC_2019_Training_Input'
        self.photo_filenames, self.labels = self.get_filenames_and_labels(data_file)
        self.process = process
        self.transform = transform

    def get_filenames_and_labels(self, csv_data_file):
        data = pd.read_csv(csv_data_file)
        filenames_list = data['image'].tolist()
        photo_filenames = []
        photo_labels = []

        MEL = data['MEL'].to_numpy()
        NV = data['NV'].to_numpy()
        BCC = data['BCC'].to_numpy()
        AK = data['AK'].to_numpy()
        BKL = data['BKL'].to_numpy()
        DF = data['DF'].to_numpy()
        VASC = data['VASC'].to_numpy()
        SCC = data['SCC'].to_numpy()
        UNK = data['UNK'].to_numpy()
        
        #removed UNK class
        class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

        photo_labels = np.vstack((MEL, NV, BCC, AK, BKL, DF, VASC, SCC))
        photo_labels = photo_labels.transpose()
        
        photo_labels = np.argmax(photo_labels, axis=1)
        
        for filename in filenames_list:
            filename += '.jpg'
            path = os.path.join(self.data_directory, filename)
            photo_filenames.append(path)
        
        return photo_filenames, photo_labels

    def __len__(self):
        return len(self.photo_filenames)

    def __getitem__(self, index):
        # if self.process == 'train' or self.process == 'validation':
        #     img_name = self.photo_filenames[idx]
        #     label = self.labels[idx]
        
        # elif self.process == 'validation':
        #     img_name = self.validation_photo_filenames[idx]
        #     label = self.validation_labels[idx]

        # elif self.process == 'test':
        #     img_name = self.photo_filenames[idx]
        #     img = Image.open(img_name).convert('RGB')
        #     img = self.transform(img)

        #     return img, img_name

        img_name = self.photo_filenames[index]
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)

        
        label = self.labels[index]



        return img, label, img_name


def load_data(dataset='ISIC', phase='train', batch_size=32, num_workers=4, shuffle=True):

    transform = data_transforms[phase]

    print('Use data transformation:', transforms)

    if dataset not in ['ISIC', 'MedMNIST']:
        raise ValueError('Dataset not implemented')

    if dataset == 'ISIC':
        _dataset = ISICDataset(_ISIC_DATA_FILE, transform=transform)
    elif dataset == 'MedMNIST':
        _dataset = 'TO BE IMPLEMENTED'
    
    return DataLoader(dataset=_dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers)



def get_validation_filenames_and_labels(num_per_class):
    #get same number of photos per class, given by num_per_class
    data = pd.read_csv(_DATA_FILE)
    filenames_list = data['image'].tolist()
    photo_filenames = []
    photo_labels = []

    MEL = data['MEL'].to_numpy()
    NV = data['NV'].to_numpy()
    BCC = data['BCC'].to_numpy()
    AK = data['AK'].to_numpy()
    BKL = data['BKL'].to_numpy()
    DF = data['DF'].to_numpy()
    VASC = data['VASC'].to_numpy()
    SCC = data['SCC'].to_numpy()

    class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    photo_labels = np.vstack((MEL, NV, BCC, AK, BKL, DF, VASC, SCC))

    photo_labels = photo_labels.transpose()
    reduced_labels = np.argmax(photo_labels, axis=1)

    for filename in filenames_list:
        filename += '.jpg'
        path = os.path.join(_DATA_DIRECTORY, filename)
        photo_filenames.append(path)
    
    validation_photo_filenames = []
    validation_labels = []

    class_count = [0 for i in range(len(class_names))]

    for i in range(len(reduced_labels)):
        label = reduced_labels[i]

        if class_count[label] < num_per_class:
            validation_photo_filenames.append(photo_filenames[i])
            validation_labels.append(reduced_labels[i])
            class_count[label] += 1
        
        if sum(class_count) >= num_per_class * len(class_names):
            break

    return validation_photo_filenames, validation_labels
        


def get_mean_and_std(photo_filenames):

    r_mean_list = np.zeros(len(photo_filenames))
    g_mean_list = np.zeros(len(photo_filenames))
    b_mean_list = np.zeros(len(photo_filenames))

    r_std_list = np.zeros(len(photo_filenames))
    g_std_list = np.zeros(len(photo_filenames))
    b_std_list = np.zeros(len(photo_filenames))

    for i in range(len(photo_filenames)):
        photo = cv2.imread(photo_filenames[i])
        norm_image = cv2.normalize(photo, None, alpha=0, beta=1, 
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
       
        b_mean = np.mean(norm_image[:,:,0])
        g_mean = np.mean(norm_image[:,:,1])
        r_mean = np.mean(norm_image[:,:,2])
        
        b_std = np.std(norm_image[:,:,0])
        g_std = np.std(norm_image[:,:,1])
        r_std = np.std(norm_image[:,:,2])

        r_mean_list[i] = r_mean
        g_mean_list[i] = g_mean
        b_mean_list[i] = b_mean

        r_std_list[i] = r_std
        g_std_list[i] = g_std
        b_std_list[i] = b_std

    rgb_mean = [np.mean(r_mean_list), np.mean(g_mean_list), np.mean(b_mean_list)]
    rgb_std = [np.mean(r_std_list), np.mean(g_std_list), np.mean(b_std_list)]

    return rgb_mean, rgb_std

