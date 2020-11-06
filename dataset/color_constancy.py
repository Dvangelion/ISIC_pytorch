import pandas as pd
import glob
import numpy
import cv2
import os


# The data directory store test images
_DATA_DIRECTORY = 'ISIC_2019_Test_Input'

# The data directory to store preprocessed images
_SAVE_DIRECTORY = 'Preprocessed_ISIC_2019_Test_Input'

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

def get_filenames_and_save_path():
    photo_filenames = glob.glob(os.path.join(_DATA_DIRECTORY, '*.jpg'))
    save_paths = []

    for filename in photo_filenames:
        photo_name = filename.split('/')[1]
        save_path = os.path.join(_SAVE_DIRECTORY, photo_name)
        save_paths.append(save_path)
    return photo_filenames, save_paths


def preprocess():
    photo_filenames, save_paths = get_filenames_and_save_path()
    assert(len(photo_filenames) == len(save_paths))

    
    for i in range(len(photo_filenames)):
        if i % 2000 == 0:
            print('processed %d images' % i)
        photo = cv2.imread(photo_filenames[i])
        preprocessed_photo = color_constancy(photo)
        cv2.imwrite(save_paths[i], preprocessed_photo)


if __name__ == '__main__':
    preprocess()

