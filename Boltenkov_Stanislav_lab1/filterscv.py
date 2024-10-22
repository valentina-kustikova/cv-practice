import cv2 as cv
import numpy as np

def conversionToGray(img):
  
    blue, green, red, = np.split(img, 3, axis = 2)
    return (0.114 * red +  0.587 * green + 0.299 * blue).astype("uint8")

def resizeImage(img, coeficents):
    
     nheight = int(img.shape[0] * coeficents[0])
     nwidth = int(img.shape[1] * coeficents[1])
     
     newIm = np.zeros((nheight, nwidth, img.shape[2]), np.uint8)   
       
     k1 = img.shape[0] / nheight
     k2 = img.shape[1] / nwidth
     
     for i in range(nheight):
        for j in range(nwidth):
            newIm[i][j] = img[int(i * k1)][int(j * k2)]   
     return newIm

def sepiaPhotoEffect(img):
  
    blue, green, red = np.split(img, 3, axis = 2)
    newBlue = np.clip(0.272 * red + 0.534 * green + 0.131 * blue, 0, 255)
    newGreen = np.clip(0.349 * red + 0.686 * green + 0.168 * blue, 0, 255)
    newRed = np.clip(0.393 * red + 0.769 * green + 0.189 * blue, 0, 255)
    
    return np.concatenate([newBlue, newGreen, newRed], axis = 2).astype("uint8")
   

def vignettePhotoEffect(img, r):
    
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = (X - (img.shape[1] // 2)) ** 2 + (Y - (img.shape[0] // 2)) ** 2
    mask = np.full((img.shape[0], img.shape[1]), r * r, np.float64)
    mask = np.clip((mask / (dist + 1)) ** 3, 0, 1)
    
    return (img * mask[::,::, np.newaxis]).astype(np.uint8)

def pixelation(img, coordinates, pix):

    newIm = img
    x0, y0, x1, y1 = coordinates
    countAreaX = (x1 - x0 + pix - 1) // pix
    countAreaY = (y1 - y0 + pix - 1) // pix
    
    for i in range(countAreaX):
        for j in range(countAreaY):
            stx, enx = x0 + i * pix, min(x0 + pix * (i + 1), x1)
            sty, eny = y0 + j * pix, min(y0 + pix * (j + 1), y1)
            newIm[stx : enx, sty : eny] = np.mean(np.mean(img[stx : enx, sty : eny], axis = 0), axis = 0) 
    return newIm