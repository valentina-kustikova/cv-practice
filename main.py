import cv2 as cv
import lab1_filters as lf
import argparse as arg

def arg_parse():
    parser = arg.ArgumentParser(description='lab1_filters')
    parser.add_argument("-ip", "--image_path", help='Input image path')
    parser.add_argument("-sz", "--desired_sizes", nargs=2, type=int, help='Input desired sizes to crop image (X,Y)')
    parser.add_argument("-r", "--vignette_radius", type=int, help='Input vignette radius')
    parser.add_argument("-px", "--pixelization_sizes", nargs=5, type=int, help='Input pixelization sizes (pixel_size, x1,y1,x2,y2)')
    args=parser.parse_args()
    return args

def main():
    args=arg_parse()
    img = cv.imread(args.image_path)
    cv.imshow("Image", img)
    cv.imshow("Gray Image", lf.toShapesOfGray(img))
    cv.imshow("Resized Image", lf.changeSize(img, args.desired_sizes[0],args.desired_sizes[1]))
    cv.imshow("Sepia Image", lf.toSepia(img))
    cv.imshow("Vignette Image", lf.vignette(img, args.vignette_radius))
    cv.imshow("Pixelizied Image", lf.pixelization(img, args.pixelization_sizes[0], args.pixelization_sizes[1], args.pixelization_sizes[2], args.pixelization_sizes[3], args.pixelization_sizes[4]))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
    
#    python main.py -ip 12.jpeg -sz 200 200 -r 250 -px 12 0 0 400 400