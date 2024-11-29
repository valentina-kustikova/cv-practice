import cv2 as cv
import numpy as np
import random

def toShapesOfGray(dst):
    
    # dst -исходное изображение
    # Вывод функции - исходное изображение в оттенках серого
    
    b,g,r=cv.split(dst)
    gray = 0.299*r + 0.587*g + 0.114*b
    gray[gray>255]=255
    return gray.astype(np.uint8)

def changeSize(dst, desired_height, desired_width):
    
    # dst - исходное изображение
    # desired_height, desired_width - новые размеры
    # Вывод функции - измененное исходное изображение
    
    height, width, channels = dst.shape
    X = np.linspace(0,width-1,desired_width)
    Y = np.linspace(0,height-1,desired_height)
    newX, newY=np.meshgrid(X,Y)
    newX=newX.astype(int)
    newY=newY.astype(int)
    resized=dst[newY,newX]
    return resized.astype(np.uint8)

def toSepia(dst):
    
    # dst - исходное изображение
    # Вывод функции - сепия исходного изображения
    
    b1,g1,r1=cv.split(dst)
    b=0.272*r1 + 0.534*g1 + 0.131*b1
    g=0.349*r1 + 0.686*g1 + 0.168*b1
    r=0.393*r1 + 0.769*g1 + 0.189*b1
    sepia=cv.merge((b,g,r))
    sepia[sepia>255]=255
    return sepia.astype(np.uint8)

def vignette(dst,r):

    # dst - исходное изображение
    # r - радиус незатемненной области
    # Вывод функции - фильтр виньетки на исходном изображении

    height, width, channels = dst.shape
    vign = np.zeros((height,width,3))
    X,Y=np.meshgrid(np.arange(width),np.arange(height))
    distances=np.sqrt((Y-height//2)*(Y-height//2)+(X-width//2)*(X-width//2))
    norm_distances = distances/np.max(distances);
    norm_r=r/np.max(distances)
    vign[:,:,0]=dst[:,:,0]*np.exp(-(norm_distances/norm_r)**4)
    vign[:,:,1]=dst[:,:,1]*np.exp(-(norm_distances/norm_r)**4)
    vign[:,:,2]=dst[:,:,2]*np.exp(-(norm_distances/norm_r)**4)
    return vign.astype(np.uint8)

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
        random.shuffle(tmp[i])
    for i in range(0,ky):
        tmp[i]=np.hstack(tmp[i])
    dst[y1:(y1+pix_size*ky),x1:(x1+pix_size*kx)]=np.vstack(tmp)
    return dst