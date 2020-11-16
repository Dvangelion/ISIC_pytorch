import pandas as pd
import argparse
import glob
import numpy 
import cv2
import os

from pandas.io import parsers

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str)

args = parser.parse_args()
phase = args.phase

# The data directory store test images
data_directory_dict = {
    'train': 'ISIC_2019_Training_Input',
    'test': 'ISIC_2019_Test_Input'
}
# TRAIN_DATA_DIRECTORY = 'ISIC_2019_Training_Input'
# TEST_DATA_DIRECTORY = 'ISIC_2019_Test_Input'

# The data directory to store preprocessed images
save_directory_dict = {
    'train': 'Preprocessed_ISIC_2019_Training_Input',
    'test': 'Preprocessed_ISIC_2019_Test_Input'
}
# TRAIN_SAVE_DIRECTORY = 'Preprocessed_ISIC_2019_Training_Input'
# TEST_SAVE_DIRECTORY = 'Preprocessed_ISIC_2019_Test_Input'

def color_constancy(img, power=6, gamma=None):
    """
    Parameters
    ----------
    img: 2D numpy array
        The original image with format of (h, w, c)
    power: int
        The degree of norm, 6 is used in reference paper
    gamma: float
        The value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = numpy.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255*pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = numpy.power(img, power)
    rgb_vec = numpy.power(numpy.mean(img_power, (0,1)), 1/power)
    rgb_norm = numpy.sqrt(numpy.sum(numpy.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*numpy.sqrt(3))
    img = numpy.minimum(numpy.multiply(img, rgb_vec), 255)
    
    return img.astype(img_dtype)

def cropping(img):
    #remove white boarder if exists
    sample_point = img[0,0,:]
    if numpy.any(sample_point > 217):
        img = img[1:-1,1:-1,:]
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret1,th1 = cv2.threshold(img_grey,35,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if w <= 200 or h <= 200:
            continue
        else:
            cropped_img = img[y:y+h, x:x+w,:]
            return cropped_img
        
    return img

def get_filenames_and_save_path(phase):
    
    data_directory = data_directory_dict[phase]
    save_directory = save_directory_dict[phase]

    photo_filenames = glob.glob(os.path.join(data_directory, '*.jpg'))
    save_paths = []

    for filename in photo_filenames:
        photo_name = filename.split('/')[1]
        save_path = os.path.join(save_directory, photo_name)
        save_paths.append(save_path)
    return photo_filenames, save_paths


def preprocess(phase):
    photo_filenames, save_paths = get_filenames_and_save_path(phase)
    assert(len(photo_filenames) == len(save_paths))

    
    for i in range(len(photo_filenames)):
        if i % 2000 == 0:
            print('processed %d images' % i)
        photo = cv2.imread(photo_filenames[i])
        cropped_photo = cropping(photo)
        preprocessed_photo = color_constancy(cropped_photo)
        cv2.imwrite(save_paths[i], preprocessed_photo)


if __name__ == '__main__':
    preprocess(phase)

