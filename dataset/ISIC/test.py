import cv2
import matplotlib.pyplot as plt

def cropping(img):
    cv2.imshow('original', img)
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret1,th1 = cv2.threshold(img_grey,35,255,cv2.THRESH_BINARY)

    cv2.imshow('th1', th1)
    #cv2.waitKey(0)

    contours,hierarchy = cv2.findContours(th1, 1, 2)
    cnt = contours[1]
    x,y,w,h = cv2.boundingRect(cnt)
    print((x,y,w,h))

    if w <= 100 and h <= 100:
        return img
    
    else:
        cropped_img = img[y:y+h-1,x:x+w-1,:]
        return cropped_img


image = cv2.imread('ISIC_2019_Training_Input/ISIC_0001133_downsampled.jpg')
cropped = cropping(image)

cv2.imshow('cropped', cropped)
cv2.waitKey(0)