import pywt
import cv2
import numpy as np

# Fusing coefficients according to the fusion method
def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef

# Params
FUSION_METHOD = 'mean' # Can be 'min' || 'max || 'on any customised operation'

# Read the two image
def Fusion(image1,image2,wavelet):
    I1 = image1
    I2 = image2
    
## Fusion algo

# First: Do wavelet transform on each image through defined wavelet
    cooef1 = pywt.wavedec2(I1[:,:], wavelet)
    cooef2 = pywt.wavedec2(I2[:,:], wavelet)
# Second:   for each level in both the images, fuse the coefficients according to the desired option
    fusedCooef = []
    for i in range(len(cooef1)-1):
        # The first values in each decomposition are the apprximation values of the top level
        if(i == 0):
            fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))
        else:
            c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],FUSION_METHOD)
            c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
            c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)

            fusedCooef.append((c1,c2,c3))

# Third: Transfer the fused coefficients back to image
    fusedImage = pywt.waverec2(fusedCooef, wavelet)

# Forth: normmalize the values to be in uint8
    fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
    fusedImage = fusedImage.astype(np.uint8)
    return fusedImage
