import os
import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_DATA_FILE = 'ISIC_2019_Training_GroundTruth.csv'


data = pd.read_csv(_DATA_FILE)

class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

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

mel_number = sum(data['MEL'])
nv_number = (sum(data['NV']))
bcc_number = (sum(data['BCC']))
ak_number = (sum(data['AK']))
bkl_number = (sum(data['BKL']))
df_number = (sum(data['DF']))
vasc_number = (sum(data['VASC']))
scc_number = (sum(data['SCC']))
unk_number = (sum(data['UNK']))


sum_numbers = [mel_number, nv_number, bcc_number, ak_number, bkl_number, df_number, vasc_number, scc_number]
print(sum_numbers)

x = np.arange(len(sum_numbers))


print(sum(data['MEL'])/25531)
print(sum(data['NV'])/25531)
print(sum(data['BCC'])/25531)
print(sum(data['AK'])/25531)
print(sum(data['BKL'])/25531)
print(sum(data['DF'])/25531)
print(sum(data['VASC'])/25531)
print(sum(data['SCC'])/25531)
print(sum(data['UNK'])/25531)


fig, ax = plt.subplots()
rects = ax.bar(x, sum_numbers)
ax.set_xticks(x)
ax.set_xticklabels(class_names)
plt.ylim([0,14000])
plt.title('Training samples by skin lesion categories')
plt.ylabel('Samples')
#ax.legend()

def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    ax.annotate('{}'.format(int(height)),
    xy=(rect.get_x()+rect.get_width()/2, height),
    xytext=(0,2),
    textcoords="offset points",
    ha='center', va='bottom')

autolabel(rects)

plt.show()
