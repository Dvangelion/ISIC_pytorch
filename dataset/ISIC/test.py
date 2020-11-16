import cv2
import numpy
import matplotlib.pyplot as plt

def cropping(img):
    sample_point = img[0,0,:]
    print(sample_point)
    if numpy.any(sample_point > 217):
        img = img[1:-1,1:-1,:]
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret1,th1 = cv2.threshold(img_grey,35,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
        if w <= 200 or h <= 200:
            continue
        else:
            cropped_img = img[y:y+h, x:x+w,:]
            return cropped_img
    
    cv2.imshow('img', img)
        
    return img


#image = cv2.imread('ISIC_2019_Training_Input/ISIC_0001133_downsampled.jpg')
image = cv2.imread('ISIC_2019_Training_Input/ISIC_0000036_downsampled.jpg')
cropped = cropping(image)

cv2.imshow('cropped', cropped)
cv2.waitKey(0)