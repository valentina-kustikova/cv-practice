import numpy as np
import cv2
import os
from typing import Optional
from parser import read_image, parse_arguments
from filters import apply_vignette, pixelate_region, convert_to_grayscale, resize_image, apply_sepia

def main() -> None:
    args = parse_arguments()
    image = read_image(args.image_path)
    result_dir = os.path.join(os.path.dirname(args.image_path), "result")
    os.makedirs(result_dir, exist_ok=True)
    
    filters = {"vignette": (apply_vignette, {"radius": args.radius, "intensity": args.intensity}),
               "grayscale": (convert_to_grayscale, {}),
               "resize": (resize_image, {"new_width": args.resize_width, "new_height": args.resize_height}),
               "sepia": (apply_sepia, {})
    }
    
    if args.filter == "pixelate":
        print("Select a region to pixelate by clicking and dragging the mouse.")
        region = cv2.selectROI("Select Region", image, showCrosshair=False)
        cv2.destroyWindow("Select Region")
        if not any(region):
            print("No region selected. Exiting.")
            return
        filtered_image = pixelate_region(image, region, args.pixel_size)
    
    elif args.filter in filters:
        func, kwargs = filters[args.filter]
        filtered_image = func(image, **kwargs)
    else:
        print("Unknown filter.")
        return
    
    output_path = os.path.join(result_dir, f"{args.filter}_" + os.path.basename(args.image_path))
    
    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image", filtered_image)
    
    if not cv2.imwrite(output_path, filtered_image):
        print(f"Error saving image {output_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()