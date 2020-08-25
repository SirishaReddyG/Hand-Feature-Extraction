import cv2
import numpy as np
from BICUBIC import Bicubic
from CANNY import Canny
from I_SOBEL import I_sobel
from FUSION import Fusion

#Get the RGB image

original_image=cv2.imdecode(np.fromfile(r'C:\Users\curio\Documents\code\FINAL\B.jpg',dtype=np.uint8),cv2.IMREAD_UNCHANGED)
print('Original Image Loaded!')
print('Shape of Original Image -', original_image.shape)
cv2.imshow('original_image',original_image)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows

# Convert the RGB image to Gray Scale

gray_image=cv2.cvtColor(original_image,cv2.COLOR_RGB2GRAY)
print('Image is converted from RGB to Gray Scale!')
cv2.imshow('gray_image',gray_image)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows

# Scale factor
ratio = 2
# Coefficient
a = -1/2

#Perorm Bicubic Interpolation

bicubic_image=Bicubic(gray_image,ratio,a)
print('Shape of Bicubic_image - ', bicubic_image.shape)
cv2.imshow('bicubic_image',bicubic_image)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows

enlarged_image=cv2.resize(gray_image,bicubic_image.shape)
print('Image is enlarged!')
print('Shape of enlarged image - ', enlarged_image.shape)
cv2.imshow('enlarged_image',enlarged_image)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows

#Get the Canny edges

canny_image=Canny(bicubic_image)
print('Shape of Canny image - ', canny_image.shape)
cv2.imshow('canny_image',canny_image)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows

#Get the Improved Sobel edges
# Pass on the enlarged_image to get the output image of size equal to canny edge
#(Facilitates Fusion of images)

i_sobel_image=I_sobel(enlarged_image)
print('Shape of i_sobel image - ', i_sobel_image.shape)
cv2.imshow('i_sobel_image',i_sobel_image)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows

# Fuse the Improved Sobel and Canny edges through DWT using haar wavelet

wavelet='haar'  # Use haar wavelet
Fused_image=Fusion(i_sobel_image,canny_image,wavelet)
print('Shape of Fused image - ', Fused_image.shape)
cv2.imshow('Fused_image',Fused_image)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows



