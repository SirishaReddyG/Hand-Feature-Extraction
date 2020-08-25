import pywt
import cv2
import numpy as np

# Fusing coefficients according to the fusion method
def Fuse_Coefficients(coeff1, coeff2, method):

    if (method == 'mean'):
        coeff = (coeff1 + coeff2) / 2
    elif (method == 'min'):
        coeff = np.minimum(coeff1,coeff2)
    elif (method == 'max'):
        coeff = np.maximum(coeff1,coeff2)
    else:
        coeff = []

    return coeff

# Params
Fusion_method = 'mean' # Can be 'min' || 'max || 'on any customised operation'

# Read the two image
def Fusion(image1,image2,wavelet):
    I1 = image1
    I2 = image2
    
## Fusion algorithm

# Step 1: Do wavelet transform on each image through defined wavelet
    coeff1 = pywt.wavedec2(I1[:,:], wavelet)
    coeff2 = pywt.wavedec2(I2[:,:], wavelet)
# Step 2:   for each level in both the images, fuse the coefficients according to the desired option
    Fused_coeff = []
    for i in range(len(coeff1)-1):
        # The first values in each decomposition are the apprximation values of the top level
        if(i == 0):
            Fused_coeff.append(Fuse_Coefficients(coeff1[0],coeff2[0],Fusion_method))
        else:
            c1 = Fuse_Coefficients(coeff1[i][0],coeff2[i][0],Fusion_method)
            c2 = Fuse_Coefficients(coeff1[i][1], coeff2[i][1], Fusion_method)
            c3 = Fuse_Coefficients(coeff1[i][2], coeff2[i][2], Fusion_method)

            Fused_coeff.append((c1,c2,c3))

# Step 3: Transfer the fused coefficients back to image
    Fused_Image = pywt.waverec2(Fused_coeff, wavelet)

# Step 4: normmalize the values to be in uint8
    Fused_Image = np.multiply(np.divide(Fused_Image - np.min(Fused_Image),(np.max(Fused_Image) - np.min(Fused_Image))),255)
    Fused_Image = Fused_Image.astype(np.uint8)
    return Fused_Image
