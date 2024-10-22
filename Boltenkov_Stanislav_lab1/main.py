import filterscv
import cv2 as cv
import numpy as np
import argparse as arg

def argumentParse():
     parser = arg.ArgumentParser()
     
     parser.add_argument(
         "-p", "--path",
         help = "Path to an image",
         required = True
     )
     parser.add_argument(
         "-s", "--store",
         help = "Path to store results",
         required = True
     )
     parser.add_argument(
         "-m", "--mode",
         type = int,
         help = "mode process image (0, 1, 2, 3, 4)",
         required = True
     )
     parser.add_argument(
         "-k",
         type = float,
         nargs = 2,
         help = "coeficents x, y for resize image"
     )
     parser.add_argument(
         "-r", "--radius",
         type = int,
         help = "radius for vignette photo effect"
     )
     parser.add_argument(
         "-c", "--coordinates",
         type = int,
         nargs= 4,
         help = "rectangle coordinates for pixelation"
     )
     parser.add_argument(
         "-px",
         type = int,
         help = "pixel size for pixelation"
     )
     
     args = parser.parse_args()
     return args
     

def main():
    
    args = argumentParse()
    img = cv.imread(args.path, -1)
    imgRes = 0

    if (args.mode == 0):
        imgRes = filterscv.conversionToGray(img)
        cv.imshow("conversionToGray", imgRes)
        
    elif (args.mode == 1):
        imgRes = filterscv.resizeImage(img, args.k)
        cv.imshow("resizeImage", imgRes)
        
    elif (args.mode == 2):
        imgRes = filterscv.sepiaPhotoEffect(img)
        cv.imshow("sepiaPhotoEffect", imgRes)
        
    elif (args.mode == 3):
        imgRes = filterscv.vignettePhotoEffect(img, args.radius)
        cv.imshow("vignettePhotoEffect", imgRes)
        
    elif (args.mode == 4):
        imgRes = filterscv.pixelation(img, args.coordinates, args.px)
        cv.imshow("pixelation", imgRes)
    
    cv.imwrite(args.store, imgRes)
    cv.waitKey(0)
    cv.destroyAllWindows()
        
main()