import argparse
import numpy as np
import os
import cv2

from filters import (
    ResizeFilter,
    GrayscaleFilter,
    AntiqueFilter,
    FadeFilter,
    FilmFilter,
    MatteFilter,
    DamageFilter,
    NeonFilter
)

def main():
    parser = argparse.ArgumentParser(
        description = "App for modifying images with filters"
    )
    parser.add_argument("--input", required = True, help = "Path to original image")
    
    parser.add_argument("--filter", required = True,
                        choices = [
                            "resize",
                            "grayscale",
                            "antique",
                            "fade",
                            "film",
                            "matte",
                            "damage",
                            "neon"
                        ], help = "Filter name")
    
    parser.add_argument("--output", required = True, help = "Path to modified image")
    
    parser.add_argument("--width", type = int, help = "Image width")
    parser.add_argument("--height", type = int, help = "Image height")
    
    parser.add_argument("--fade_factor", type = float, default = 0.7, 
                        help = "Fade-factor for FadeFilter")
    parser.add_argument("--fade_brightness", type = float, default = 40,
                        help = "Brightness for FadeFilter")
    
    parser.add_argument("--red_factor", type = float, default = 1.5,
                        help = "Red factor for FilmFilter")
    parser.add_argument("--blue_factor", type = float, default = 0.7,
                        help = "Blue factor for FilmFilter")
    
    parser.add_argument("--radius", type = float, default = 0.5, 
                        help = "Radius for MatteFilter")
    parser.add_argument("--softness", type = float, default = 0.8, 
                        help = "Softness for MatteFilter")
    
    parser.add_argument("--grain", type = int, default = 8, 
                        help = "Grain for DamageFilter")
    parser.add_argument("--scratch_amount", type = int, default = 55, 
                        help = "Scratch amount for DamageFilter")
    parser.add_argument("--dust_amount", type = int, default = 110, 
                        help = "Dust amount for DamageFilter")
    
    parser.add_argument("--intensity", type = float, default = 1.5, 
                        help = "Intensity for NeonFilter")
    parser.add_argument("--neon_brightness", type = float, default = 1.0,
                        help = "Brightness for NeonFilter")
    
    args = parser.parse_args()
    
    image = cv2.imdecode(np.fromfile(args.input, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Could not load image: {args.input}")
    
    if args.filter == "resize":
        if args.width is None or args.height is None:
            parser.error("For this filter width and height are required")
        image_filter = ResizeFilter(args.width, args.height)
        
    elif args.filter == "grayscale":
        image_filter = GrayscaleFilter()
    
    elif args.filter == "antique":
        image_filter = AntiqueFilter()
        
    elif args.filter == "fade":
        image_filter = FadeFilter(args.fade_factor, args.fade_brightness)
        
    elif args.filter == "film":
        image_filter = FilmFilter(args.red_factor, args.blue_factor)
    
    elif args.filter == "matte":
        image_filter = MatteFilter(args.radius, args.softness)
        
    elif args.filter == "damage":
        image_filter = DamageFilter(args.grain, args.scratch_amount, args.dust_amount)
        
    elif args.filter == "neon":
        image_filter = NeonFilter(args.intensity, args.neon_brightness)
        
    result = image_filter.apply_filter(image)
    
    try:
        _, ext = os.path.splitext(args.output)
        success, encoded_img = cv2.imencode(ext, result)
        if success:
            encoded_img.tofile(args.output)
        else:
            success = False
    except Exception:
        success = False

    if not success:
        parser.error(f"Could not save image: {args.output}")
    
    print(f"Image saved: {args.output}")

if __name__ == "__main__":
    main()
    


