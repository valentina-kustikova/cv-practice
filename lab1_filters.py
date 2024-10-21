import cv2 as cv
import numpy as np
import random

def toShapesOfGray(dst):
    
    # dst -исходное изображение
    # Вывод функции - исходное изображение в оттенках серого
    
    b,g,r=cv.split(dst)
    gray = np.uint8(0.299*r) + np.uint8(0.587*g) + np.uint8(0.114*b)
    return gray

def changeSize(dst, desired_height, desired_width):
    
    # dst - исходное изображение
    # desired_height, desired_width - новые размеры
    # Вывод функции - измененное исходное изображение
    
    resized = np.zeros((desired_height,desired_width,3),np.uint8)
    stepx=dst.shape[1]/desired_width
    stepy=dst.shape[0]/desired_height
    for i in range (0, desired_width):
        for j in range (0, desired_height):
            resized[j,i]=dst[int(j*stepy),int(i*stepx)]
    return resized

def toSepia(dst):
    
    # dst - исходное изображение
    # Вывод функции - сепия исходного изображения
    
    b1,g1,r1=cv.split(dst)
    b=np.uint8(0.272*r1) + np.uint8(0.534*g1) + np.uint8(0.131*b1)
    g=np.uint8(0.349*r1) + np.uint8(0.686*g1) + np.uint8(0.168*b1)
    r=np.uint8(0.393*r1) + np.uint8(0.769*g1) + np.uint8(0.189*b1)
    sepia=cv.merge((b,g,r))
    return sepia

def vignette(dst,r): # Работает сомнительно

    # dst - исходное изображение
    # r - радиус незатемненной области
    # Вывод функции - фильтр виньетки на исходном изображении

    height, width, channels = dst.shape
    vign = dst
    mask = np.empty((height,width,3), np.uint8)
    for i in range (0,height//2):
        mask[i] = mask[height-i-1] = i
    gray =toShapesOfGray(dst)
    mask = cv.circle(mask, (width//2,height//2), r, (255,255,255),thickness=-1)
    mask = cv.bitwise_and(mask, dst)
    return mask

def pixelization(dst,pix_size,x1,y1,x2,y2): 
    
    # dst - исходное изображение
    # pix_size - размер пикселя
    # x1, y1 - левый верхний угол пикселизируемой области исходного изображения
    # x2, y2 - правый нижний угол пикселизируемой области исходного изображения
    # Вывод функции - исходное изображение с пикселизированной заданной областью
    
    kx=(abs(x1-x2)//pix_size)
    ky=(abs(y1-y2)//pix_size)
    pixpart=dst[y1:(y1+pix_size*ky),x1:(x1+pix_size*kx)]
    tmp = np.vsplit(pixpart, ky)
    for i in range(0,ky):
        tmp[i]=np.hsplit(tmp[i], kx)
    for i in range(0,ky):
        for j in range(0,kx):
            tmp[i][j]=tmp[random.randint(0,ky-1)][random.randint(0,kx-1)]
    for i in range(0,ky):
        tmp[i]=np.hstack(tmp[i])
    tmp=np.vstack(tmp)
    dst[y1:(y1+pix_size*ky),x1:(x1+pix_size*kx)]=tmp
    return dst