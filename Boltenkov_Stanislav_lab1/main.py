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
         "-m", "--mode",
         type = int,
         help = "mode process image (0, 1, 2, 3, 4)",
         required = True
     )
     parser.add_argument(
         "-kx",
         type = float,
         help = "coeficent x for resize image"
     ) 
     parser.add_argument(
         "-ky",
         type = float,
         help = "coeficent y for resize image"
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

    if (args.mode == 0):
        cv.imshow("conversionToGray", filterscv.conversionToGray(img))
        
    elif (args.mode == 1):
        
        cv.imshow("resizeImage", filterscv.resizeImage(img, args.kx, args.ky))
        
    elif (args.mode == 2):
        cv.imshow("sepiaPhotoEffect", filterscv.sepiaPhotoEffect(img))
        
    elif (args.mode == 3):
        cv.imshow("vignettePhotoEffect", filterscv.vignettePhotoEffect(img, args.radius))
        
    elif (args.mode == 4):
        cv.imshow("pixelation", filterscv.pixelation(img, args.coordinates[0], args.coordinates[1], args.coordinates[2], args.coordinates[3], args.px))
        
    cv.waitKey(0)
    cv.destroyAllWindows()
        
main()