import cv2
import numpy as np
import math

# kernel for Interpolation
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

# Paddnig
def padding(img,H,W):
    print('Shape of the image before padding -', img.shape)
    p_img = np.zeros((H+4,W+4))
    p_img[2:H+2,2:W+2] = img
    #Pad the first/last two cols and rows
    p_img[2:H+2,0:2]=img[:,0:1]
    p_img[H+2:H+4,2:W+2]=img[H-1:H,:]
    p_img[2:H+2,W+2:W+4]=img[:,W-1:W]
    p_img[0:2,2:W+2]=img[0:1,:]
    #Pad the missing eight points
    p_img[0:2,0:2]=img[0,0]
    p_img[H+2:H+4,0:2]=img[H-1,0]
    p_img[H+2:H+4,W+2:W+4]=img[H-1,W-1]
    p_img[0:2,W+2:W+4]=img[0,W-1]
    print('Shape of the image after padding -', p_img.shape)
    return p_img

# Bicubic operation
def Bicubic(img, ratio, a):
    #Get image size
    H,W = img.shape
    img = padding(img,H,W)
    #Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    d = np.zeros((dH, dW))
    h = 1/ratio
    print('Bicubic interpolation started!')
    print('Bicubic interpolation in progress...')
    for j in range(dH):
        for i in range(dW):
            x, y = i * h + 2 , j * h + 2

            x1 = 1 + x - math.floor(x)
            x2 = x - math.floor(x)
            x3 = math.floor(x) + 1 - x
            x4 = math.floor(x) + 2 - x

            y1 = 1 + y - math.floor(y)
            y2 = y - math.floor(y)
            y3 = math.floor(y) + 1 - y
            y4 = math.floor(y) + 2 - y

            mat_x = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
            mat_y = np.matrix([[img[int(y-y1),int(x-x1)],img[int(y-y2),int(x-x1)],img[int(y+y3),int(x-x1)],img[int(y+y4),int(x-x1)]],
                                   [img[int(y-y1),int(x-x2)],img[int(y-y2),int(x-x2)],img[int(y+y3),int(x-x2)],img[int(y+y4),int(x-x2)]],
                                   [img[int(y-y1),int(x+x3)],img[int(y-y2),int(x+x3)],img[int(y+y3),int(x+x3)],img[int(y+y4),int(x+x3)]],
                                   [img[int(y-y1),int(x+x4)],img[int(y-y2),int(x+x4)],img[int(y+y3),int(x+x4)],img[int(y+y4),int(x+x4)]]])
            mat_p = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
            d[j, i] = np.dot(np.dot(mat_x, mat_y),mat_p)
    print('Bicubic Interpolation Ended!')

    return d
