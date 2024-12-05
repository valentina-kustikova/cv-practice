import cv2
import os
from utils import read_image, select_region, parse_arguments
from filters import apply_vignette, pixelate_region, convert_to_grayscale, resize_image, apply_sepia


def main():
    args = parse_arguments()

    try:
        image = read_image(args.image_path)
        
        filtered_image = None
        title = ""
        output_path = ""

        if args.filter == 'vignette':
            filtered_image = apply_vignette(image, radius=args.radius, intensity=args.intensity)
            title = 'Vignette Effect'
            output_path = os.path.join(os.path.dirname(args.image_path), "vignette_" + os.path.basename(args.image_path))

        elif args.filter == 'pixelate':
            print("Select the region to pixelate by dragging the mouse.")
            region = select_region(image)
            if region is None:
                print("No region selected. Exiting.")
                return
            filtered_image = pixelate_region(image.copy(), region, args.pixel_size)
            title = 'Pixelated Region'
            output_path = os.path.join(os.path.dirname(args.image_path), "pixelated_" + os.path.basename(args.image_path))

        elif args.filter == 'grayscale':
            filtered_image = convert_to_grayscale(image)
            title = 'Grayscale Image'
            output_path = os.path.join(os.path.dirname(args.image_path), "grayscale_" + os.path.basename(args.image_path))

        elif args.filter == 'resize':
            if args.resize_width is None or args.resize_height is None:
                print("To resize, you must specify --resize_width and --resize_height.")
                return
            filtered_image = resize_image(image, args.resize_width, args.resize_height)
            title = 'Resized Image'
            output_path = os.path.join(os.path.dirname(args.image_path), "resized_" + os.path.basename(args.image_path))

        elif args.filter == 'sepia':
            filtered_image = apply_sepia(image)
            title = 'Sepia Effect'
            output_path = os.path.join(os.path.dirname(args.image_path), "sepia_" + os.path.basename(args.image_path))

    
        cv2.imshow('Original Image', image)
        cv2.imshow(title, filtered_image)

        cv2.imwrite(output_path, filtered_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()