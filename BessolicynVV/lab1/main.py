import sys
import numpy as np
import argparse
import cv2 as cv
import os

def resize(img, w=None, h=None, scal=None):
    src_h, src_w = img.shape[:2]
    if scal is not None:
        if (w is not None) or (h is not None):
            raise ValueError("Либо scal, либо (w/h).")
        if scal <= 0:
            raise ValueError("scal должен быть положительным числом.")
        new_h = int(round(src_h*scal))
        new_w = int(round(src_w*scal))
    else:
        if (w is not None) and (h is not None):
            new_w = w
            new_h = h
        if h is not None:
            ratio = h/src_h
            new_w = int(src_w*ratio)
            new_h = h
        if w is not None:
            ratio = w/src_w
            new_h = int(src_h*ratio)
            new_w = w
        if (w is None) and (h is None):
            return img.copy()

    if new_w == src_w and new_h == src_h:
        return img.copy()

    scal_x = src_w/new_w
    scal_y = src_h/new_h

    src_x = (np.arange(new_w, dtype=np.float32) + 0.5) * scal_x - 0.5
    src_y = (np.arange(new_h, dtype=np.float32) + 0.5) * scal_y - 0.5

    x_idx = np.rint(src_x).astype(np.int32)
    y_idx = np.rint(src_y).astype(np.int32)

    # нуу надо точно не выходить за границы
    x_idx = np.clip(x_idx, 0, src_w - 1)
    y_idx = np.clip(y_idx, 0, src_h - 1)

    if img.ndim == 2:
        out = img[y_idx[:, None], x_idx[None, :]]
    else:
        out = img[y_idx[:, None], x_idx[None, :], :]

    return out



def sepia(img, intensity: float = 1.0):
    sepia = np.array([
        [0.439, 0.859, 0.092],
        [0.398, 0.734, 0.088],
        [0.310, 0.605, 0.073]
    ])

    img_f = img.astype(np.float32)

    sep_img = img_f @ sepia.T
    t = float(np.clip(intensity, 0.0, 1.0))
    out = (1.0 - t) * img_f + t * sep_img
    out = np.clip(out, 0, 255)

    return out.astype(np.uint8)



def vignette(img, intensity: float = 1.0, radius: int = 100):
    src_h, src_w = img.shape[:2]
    cen_x, cen_y = 0.5*(src_w - 1), 0.5*(src_h - 1)

    r_max = int(radius)
    if r_max <= 0:
        r_max = 1

    y, x = np.ogrid[:src_h, :src_w]
    r = np.sqrt((x-cen_x)**2 + (y-cen_y)**2)

    m = 1.0 - np.clip(r / float(r_max), 0.0, 1.0)
    t = float(np.clip(intensity, 0.0, 1.0))

    scale = ((1.0 - t) + t * m)[..., None]

    out = img.astype(np.float32) * scale
    return np.clip(out, 0, 255).astype(np.uint8)



def pixelize(img, x: int = 0, y: int = 0, region_w: int = 0, region_h: int = 0, pix_sz: int = 1):
    res = img.copy()

    if pix_sz <= 1:
        return res
    if region_w == 0:
        region_w = img.shape[1] - x
    if region_h == 0:
        region_h = img.shape[0] - y

    ed_x = min(x + region_w, img.shape[1])
    ed_y = min(y + region_h, img.shape[0])

    for i in range(y, ed_y, pix_sz):
        for j in range(x, ed_x, pix_sz):
            block_ed_i = min(i + pix_sz, ed_y)
            block_ed_j = min(j + pix_sz, ed_x)
            block = res[i:block_ed_i, j:block_ed_j]
            if block.size > 0:
                avg_color = block.mean(axis=(0, 1)).astype(int)

                res[i:block_ed_i, j:block_ed_j] = avg_color

    return res



def add_border_frame(img, b_w, color):
    h, w = img.shape[:2]
    new_h = h + 2 * b_w
    new_w = w + 2 * b_w

    if len(img.shape) == 3:
        bordered_img = np.zeros((new_h, new_w, 3), dtype=img.dtype)
        bordered_img[:, :] = color
    else:
        bordered_img = np.zeros((new_h, new_w), dtype=img.dtype)
        bordered_img[:, :] = color[0] if isinstance(color, (list, tuple)) else color

    bordered_img[b_w:b_w+h, b_w:b_w+w] = img

    return bordered_img



def add_figure_frame(img, fr_ind, threshold=50.0):
    frame = cv.imread(f"src/frame{fr_ind}.jpg", cv.IMREAD_COLOR)
    if frame is None:
        print(f"Cannot load frame: src/frame{fr_ind}.jpg")
        return img.copy()

    h, w = img.shape[:2]
    if frame.shape[:2] != (h, w):
        frame = cv.resize(frame, (w, h))

    frame_float = frame.astype(np.float32)
    img_float = img.astype(np.float32)

    diff_B = frame_float[..., 0] - 255.0
    diff_G = frame_float[..., 1] - 255.0
    diff_R = frame_float[..., 2] - 255.0

    color_distance = np.sqrt(diff_B**2 + diff_G**2 + diff_R**2)

    mask = (color_distance > threshold).astype(np.float32)[..., np.newaxis]
    result = img_float * (1.0 - mask) + frame_float * mask
    return np.clip(result, 0, 255).astype(np.uint8)



def add_glare_effect(image, strength=0.7):
    glare = cv.imread("src/glare.jpg", cv.IMREAD_COLOR)
    if glare is None:
        print("Cannot load glare texture: src/glare.jpg")
        return image.copy()

    h, w = image.shape[:2]
    if glare.shape[:2] != (h, w):
        glare = cv.resize(glare, (w, h))

    image_float = image.astype(np.float32) / 255.0
    glare_float = glare.astype(np.float32) / 255.0
    glare_adjusted = np.clip(glare_float * strength, 0.0, 1.0)
    result = 1.0 - (1.0 - image_float) * (1.0 - glare_adjusted)
    return (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)



def add_watercolor_texture(image, intensity=0.3, texture_path="src/watercolor_paper.jpg"):
    texture = cv.imread(texture_path, cv.IMREAD_COLOR)
    if texture is None:
        print(f"Cannot load texture: {texture_path}")
        return image.copy()

    if texture.shape[:2] != image.shape[:2]:
        texture = cv.resize(texture, (image.shape[1], image.shape[0]))

    texture_gray = texture.mean(axis=2).astype(np.float32) / 255.0
    texture_mask = (1.0 - texture_gray)[..., np.newaxis]

    image_float = image.astype(np.float32)
    texture_float = texture.astype(np.float32)

    blended = (image_float * (1.0 - intensity * texture_mask) + texture_float * (intensity * texture_mask))

    return np.clip(blended, 0, 255).astype(np.uint8)



def cli_parser():
    parser = argparse.ArgumentParser(description="Image CLI")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--output", help="Path to save result")
    parser.add_argument("--show", action="store_true", help="Preview result")
    subs = parser.add_subparsers(dest="op", required=True)

    subs.add_parser("about", help="What this program does")

    p_resize = subs.add_parser("resize", help="Resize (nearest neighbor)")
    p_resize.add_argument("--width", "-W", type=int, help="Target width")
    p_resize.add_argument("--height", "-H", type=int, help="Target height")
    p_resize.add_argument("--scale", type=float, help="Scale factor")

    p_sepia = subs.add_parser("sepia", help="Sepia filter")
    p_sepia.add_argument("--intensity", type=float, default=1.0)

    p_vign = subs.add_parser("vignette", help="Vignette effect")
    p_vign.add_argument("--intensity", type=float, default=1.0)
    p_vign.add_argument("--radius", type=float, default=100.0)

    p_pix = subs.add_parser("pixelize", help="Pixelate region")
    p_pix.add_argument("--x", type=int, default=0, help="X coordinate")
    p_pix.add_argument("--y", type=int, default=0, help="Y coordinate")
    p_pix.add_argument("--width", "-W", type=int, default=0, help="Region width")
    p_pix.add_argument("--height", "-H", type=int, default=0, help="Region height")
    p_pix.add_argument("--pixel_size", type=int, default=10, help="Pixel block size")

    p_border = subs.add_parser("add_border_frame", help="Border frame")
    p_border.add_argument("--width", "-W", type=int, required=True, help="Border width")
    p_border.add_argument("--b", type=int, default=255, help="Blue channel")
    p_border.add_argument("--g", type=int, default=255, help="Green channel")
    p_border.add_argument("--r", type=int, default=255, help="Red channel")

    p_fig = subs.add_parser("add_figure_frame", help="Decorative frame")
    p_fig.add_argument("--index", type=int, required=True, help="Frame index (0-4)")
    p_fig.add_argument("--threshold", "-t", type=float, default=50.0)

    p_glr = subs.add_parser("add_glare_effect", help="Glare blend")
    p_glr.add_argument("--strength", type=float, default=0.7)

    p_wc = subs.add_parser("add_watercolor_texture", help="Watercolor texture")
    p_wc.add_argument("--intensity", type=float, default=0.3)

    return parser.parse_args()


def main():
    args = cli_parser()

    if args.op == "about":
        print(
            "Applies image operations: resize, sepia, vignette, pixelize, frames, glare, watercolor.\n"
            "Usage: python main.py --image in.jpg OP [flags]\n"
            "Notes: resize needs --width/--height/--scale; pixelize uses (x,y) coordinates.\n"
        )
        return

    if not args.image:
        print("Error: --image is required.", file=sys.stderr)
        sys.exit(2)

    img = cv.imread(args.image, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    if args.op == "resize":
        out_img = resize(img, w=args.width, h=args.height, scal=args.scale)
    elif args.op == "sepia":
        out_img = sepia(img, args.intensity)
    elif args.op == "vignette":
        out_img = vignette(img, args.intensity, args.radius)
    elif args.op == "pixelize":
        out_img = pixelize(img, args.x, args.y, args.width, args.height, args.pixel_size)
    elif args.op == "add_border_frame":
        out_img = add_border_frame(img, args.width, (args.b, args.g, args.r))
    elif args.op == "add_figure_frame":
        out_img = add_figure_frame(img, args.index, args.threshold)
    elif args.op == "add_glare_effect":
        out_img = add_glare_effect(img, args.strength)
    elif args.op == "add_watercolor_texture":
        out_img = add_watercolor_texture(img, args.intensity)
    else:
        print("Unknown operation.", file=sys.stderr)
        sys.exit(2)

    base_name, ext = os.path.splitext(os.path.basename(args.image))
    if not ext:
        ext = ".jpg"
    if args.output:
        out_path = args.output
    else:
        os.makedirs("out", exist_ok=True)
        out_path = os.path.join("out", f"{base_name}__{args.op}{ext}")

    if not cv.imwrite(out_path, out_img):
        print(f"Save failed: {out_path}", file=sys.stderr)
        sys.exit(3)

    print(out_path)

    if args.show:
        cv.imshow(args.op, out_img)
        cv.waitKey(0)
        cv.destroyAllWindows()



if __name__ == "__main__":
    main()







