import cv2 as cv
import numpy as np

def conversionToGray(imagePath):
    img = cv.imread(imagePath, -1)
    
    grayIm = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            grayIm[i][j] = np.uint8(0.299 * img[i][j][2] + 0.587 * img[i][j][1] + 0.114 * img[i][j][0])#BGR
            
         
    cv.imshow("conversionToGray", grayIm)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def resizeImage(imagePath, kheight, kwidth):
     img = cv.imread(imagePath, -1)
     nheight = int(img.shape[0] * kheight)
     nwidth = int(img.shape[1] * kwidth)
     newIm = np.zeros((nheight, nwidth, 3), np.uint8)
     
     k1 = img.shape[0] / nheight
     k2 = img.shape[1] / nwidth
     
     for i in range(nheight):
        for j in range(nwidth):
            newIm[i][j] = img[int(i * k1)][int(j * k2)]   
            
     cv.imshow("resizeImage", newIm)
     cv.waitKey(0)
     cv.destroyAllWindows()
     return

def sepiaPhotoEffect(imagePath):
    img = cv.imread(imagePath, -1)
    
    newIm = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newIm[i][j][0] = min(0.272 * img[i][j][2] + 0.534 * img[i][j][1] + 0.131 * img[i][j][0], 255) 
            newIm[i][j][1] = min(0.349 * img[i][j][2] + 0.686 * img[i][j][1] + 0.168 * img[i][j][0], 255)
            newIm[i][j][2] = min(0.393 * img[i][j][2] + 0.769 * img[i][j][1] + 0.189 * img[i][j][0], 255)
            
    cv.imshow("sepiaPhotoEffect", newIm)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def vignettePhotoEffect(imagePath):
    img = cv.imread(imagePath, -1)
    
    newIm = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    centerx = img.shape[0] // 2
    centery = img.shape[1] // 2
    r = min(centerx, centery) + (max(centerx, centery) - min(centerx, centery)) // 2
    r = r * r
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist = (centerx - i) * (centerx - i) + (centery - j) * (centery - j)
            newIm[i][j] = img[i][j]
            if (dist > r):
                newIm[i][j] = pow(r / dist, 3) * img[i][j]
            
    cv.imshow("vignettePhotoEffect", newIm)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return

def pixelation(imagePath, x0, y0, x1, y1):
    img = cv.imread(imagePath, -1)
    
    newIm = img
    
    if (y0 > y1 or x0 > x1):
        y0, y1, x0, x1 = y1, y0, x1, x0
    
    countAreaX = (x1 - x0 + 9) // 10
    countAreaY = (y1 - y0 + 9) // 10
    
    for i in range(countAreaX):
        for j in range(countAreaY):
            stx = x0 + i * 10
            enx = min(x0 + 10 * (i + 1), x1)
            sty = y0 + j * 10
            eny = min(y0 + 10 * (j + 1), y1)
            newIm[stx : enx, sty : eny] = np.mean(np.mean(img[stx : enx, sty : eny], axis = 0), axis = 0)
        
    cv.imshow("pixelation", newIm)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return