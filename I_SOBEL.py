import cv2
import numpy as np

def I_sobel(image):
    kernel = np.ones((5,5),np.uint8)
    image=image
    #opening function
    opening=cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    #closing function
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
    #sobel operator
    sobel = cv2.Sobel(src=closing,ddepth=-1,dx=1,dy=1,ksize=3)
    return sobel
