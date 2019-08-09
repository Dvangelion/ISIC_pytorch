import os
import cv2
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


_NUM_VALIDATION = 2553
_DATA_FILE = './dataset/ISIC_2019_Training_GroundTruth.csv'
_DATA_DIRECTORY = './dataset/Preprocessed_ISIC_2019_Training_Input'

class ISICDataLoader(Dataset):
    def __init__(self, photo_filenames, photo_labels, process=None, transform=None):
        self.train_photo_filenames = photo_filenames
        self.train_labels = photo_labels
        self.validation_photo_filenames = photo_filenames[:_NUM_VALIDATION]
        self.validation_labels = photo_labels[:_NUM_VALIDATION]
        self.process = process
        self.transform = transform

    def __len__(self):
        if self.process == 'train':
            return len(self.train_labels)
        elif self.process == 'validation':
            return len(self.validation_labels)
    
    def __getitem__(self, idx):
        if self.process == 'train':
            img_name = self.train_photo_filenames[idx]
            label = self.train_labels[idx]
        
        elif self.process == 'validation':
            img_name = self.validation_photo_filenames[idx]
            label = self.validation_labels[idx]
        
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)

        return img, label

def get_filenames_and_labels():
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
    #UNK = data['UNK'].to_numpy()
    
    #removed UNK class
    class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    photo_labels = np.vstack((MEL, NV, BCC, AK, BKL, DF, VASC, SCC))
    photo_labels = photo_labels.transpose()
    photo_labels = np.argmax(photo_labels, axis=1)

    for filename in filenames_list:
        filename += '.jpg'
        path = os.path.join(_DATA_DIRECTORY, filename)
        photo_filenames.append(path)
    
    return photo_filenames, photo_labels, class_names


def get_validation_filenames_and_labels(num_per_class):
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

