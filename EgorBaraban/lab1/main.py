import numpy as np
import argparse
import cv2 as cv
import os
import sys


def gaussian_kernel1d(sigma):

    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    return k.astype(np.float32)

def convolve_separable(img_f, k):

    H, W, C = img_f.shape
    r = (len(k) - 1) // 2
    padded_h = np.pad(img_f, ((0, 0), (r, r), (0, 0)), mode='edge')
    tmp = np.empty_like(img_f)
    cols = [padded_h[:, i:i+W, :] for i in range(2*r + 1)]     
    stack = np.stack(cols, axis=0)                              
    tmp[:] = np.tensordot(k, stack, axes=([0], [0]))            
    padded_v = np.pad(tmp, ((r, r), (0, 0), (0, 0)), mode='edge')
    rows = [padded_v[i:i+H, :, :] for i in range(2*r + 1)]      
    stack = np.stack(rows, axis=0)                              
    out = np.tensordot(k, stack, axes=([0], [0]))               

    return out


def resize(img, new_h, new_w, sigma = None):
    new_h = max(int(round(new_h)), 1)
    new_w = max(int(round(new_w)), 1)
    img_f = img.astype(np.float32)
    H, W = img_f.shape[:2]

    sy = new_h / H
    sx = new_w / W

    if sigma is None:
        down = max(1.0 / max(sx, 1.0), 1.0 / max(sy, 1.0)) 
        sigma = 0.5 * (down - 1.0) if down > 1.0 else 0.0
        sigma = max(sigma, 0.0)

    if sigma > 0:
        k = gaussian_kernel1d(sigma)
        img_f = convolve_separable(img_f, k)

    dst_y = np.arange(new_h, dtype=np.float32)
    dst_x = np.arange(new_w, dtype=np.float32)
    
    src_y = (dst_y + 0.5) / sy - 0.5
    src_x = (dst_x + 0.5) / sx - 0.5
    
    src_y = np.clip(src_y, 0.0, H - 1)
    src_x = np.clip(src_x, 0.0, W - 1)
    
    y0 = np.floor(src_y).astype(np.int32)
    x0 = np.floor(src_x).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    
    wy = (src_y - y0).astype(np.float32)
    wx = (src_x - x0).astype(np.float32)
    inv_wy = 1.0 - wy
    inv_wx = 1.0 - wx
    
    top = img_f[y0, :, :]
    bot = img_f[y1, :, :]
    
    top_left  = top[np.arange(new_h)[:, None], x0[None, :], :]
    top_right = top[np.arange(new_h)[:, None], x1[None, :], :]
    bot_left  = bot[np.arange(new_h)[:, None], x0[None, :], :]
    bot_right = bot[np.arange(new_h)[:, None], x1[None, :], :]
    
    wy_ = wy[:, None, None]
    wx_ = wx[None, :, None]
    inv_wy_ = inv_wy[:, None, None]
    inv_wx_ = inv_wx[None, :, None]
    
    out = (top_left  * inv_wx_ * inv_wy_ +
           top_right *  wx_    * inv_wy_ +
           bot_left  * inv_wx_ *  wy_    +
           bot_right *  wx_    *  wy_)


    out = out.astype(img.dtype)

    return out

def sepia_filter(img, intensity = 1.0):
    img_f = img.astype(np.float32)
    sepia = np.array([[0.131, 0.534, 0.272], 
                      [0.168, 0.686, 0.349],
                      [0.189, 0.769, 0.393]], dtype=np.float32)
    full_sepia = np.empty_like(img, dtype=np.float32)
    for x in list(range(img.shape[0])):
        for y in list(range(img.shape[1])):
            for i in list(range(3)):
                full_sepia[x, y, i] = np.dot(sepia[i], img_f[x, y])
        
    
    res = (1 - intensity) * img + intensity * full_sepia
    res = np.clip(res, 0, 255)
    return res.astype(np.uint8)
    

def vignette(img, intensity=1, radius=200):
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.hypot(x - cx, y - cy)

    if radius is None:
        rmax = min(cx, cy)
    else:
        rmax = float(radius)
        if 0 < rmax <= 1:
            rmax *= min(cx, cy)
        rmax = max(rmax, 1e-6)

    m = 1.0 - np.clip(r / rmax, 0.0, 1.0) 
    scale = ((1 - intensity) + intensity * m)[..., None]

    out = img.astype(np.float32) * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def pixelate(img, left_top, right_bot, pixel_size = 10):
    img_c = img.copy()
    l_x, l_y = left_top
    r_x, r_y = right_bot
    if (l_x > r_x or l_y > r_y or l_y < 0 or l_x < 0 or r_x < 0 or r_y < 0):
        raise ValueError("Invalid bounds")
    pixelated_region = img[l_x:r_x, l_y:r_y]
    pH, pW = pixelated_region.shape[:2]
    if pixelated_region.size == 0:
        return img_c
    pixelated_region = resize(pixelated_region, pH / pixel_size, pW / pixel_size)
    pixelated_region = resize(pixelated_region, pH, pW)
    img_c[l_x:r_x, l_y:r_y] = pixelated_region
    return img_c


def add_rect_frame(img, w, color):
    if (w < 0):
        raise ValueError("Width can not be negative")
    color = np.clip(color, 0, 255)
    img_c = img.copy()
    
    img_c[0:w] = color
    img_c[-w:] = color
    img_c[w:-w, 0:w] = color
    img_c[w:-w, -w:] = color
    return img_c


def add_figure_frame(img, f_index, t=50.0):
    frame = cv.imread(f"src/frame{f_index}.jpg", cv.IMREAD_COLOR)
    if frame is None:
        print("Can not load such frame")
        return img

    h, w = img.shape[:2]
    if frame.shape[:2] != (h, w):
        frame = resize(frame, h, w)

    frm = frame.astype(np.float32)
    dB = frm[..., 0] - 255.0
    dG = frm[..., 1] - 255.0
    dR = frm[..., 2] - 255.0
    color_diff = np.hypot(np.hypot(dB, dG), dR)

    mask = (color_diff > float(t)).astype(np.float32)[..., None]

    out = img.astype(np.float32) * (1.0 - mask) + frm * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def add_glare(image, strength=0.85):
    glare = cv.imread("src/glare_cool.jpg", cv.IMREAD_COLOR)
    if glare is None:
        print("Can not load glare")
        return image

    h, w = image.shape[:2]
    if glare.shape[:2] != (h, w):
        glare = resize(glare, h, w)

    img_f = image.astype(np.float32) / 255.0
    glr_f = glare.astype(np.float32) / 255.0

    glr_f = np.clip(glr_f * float(strength), 0.0, 1.0)
    out = 1.0 - (1.0 - img_f) * (1.0 - glr_f)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def add_watercolor_texture(image, intensity=0.3):
    tex = cv.imread("src/watercolor_paper.jpg", cv.IMREAD_COLOR)
    if tex is None:
        print("Can not load texture")
        return image
    if tex.shape[:2] != image.shape[:2]:
        tex = resize(tex, image.shape[0], image.shape[1]) 

    m = (1.0 - tex.mean(axis=2).astype(np.float32) / 255.0)[..., None]

    out = image.astype(np.float32) * (1.0 - intensity * m) + tex.astype(np.float32) * (intensity * m)
    return np.clip(out, 0, 255).astype(np.uint8)




def cli_arg_parser():
    parser = argparse.ArgumentParser(description="Image CLI")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--output", help="Path to save result")
    parser.add_argument("--show", action="store_true", help="Preview result")
    subs = parser.add_subparsers(dest="op", required=True)

    subs.add_parser("about", help="What this program does")

    p_resize = subs.add_parser("resize", help="Resize (bilinear)")
    p_resize.add_argument("--height", "-H", type=int, required=True)
    p_resize.add_argument("--width", "-W", type=int, required=True)
    p_resize.add_argument("--sigma", type=float, default=None)

    p_sepia = subs.add_parser("sepia_filter", help="Sepia")
    p_sepia.add_argument("--intensity", type=float, default=1.0)

    p_vign = subs.add_parser("vignette", help="Vignette")
    p_vign.add_argument("--intensity", type=float, default=1.0)
    p_vign.add_argument("--radius", type=float, default=200.0)

    p_pix = subs.add_parser("pixelate", help="Pixelate ROI (y,x)")
    p_pix.add_argument("--y1", type=int, required=True)
    p_pix.add_argument("--x1", type=int, required=True)
    p_pix.add_argument("--y2", type=int, required=True)
    p_pix.add_argument("--x2", type=int, required=True)
    p_pix.add_argument("--block", type=int, required=True)

    p_rect = subs.add_parser("add_rect_frame", help="Rect frame (BGR)")
    p_rect.add_argument("--thickness", type=int, required=True)
    p_rect.add_argument("--b", type=int, default=0)
    p_rect.add_argument("--g", type=int, default=0)
    p_rect.add_argument("--r", type=int, default=0)

    p_fig = subs.add_parser("add_figure_frame", help="Decorative frame")
    p_fig.add_argument("--index", type=int, required=True)
    p_fig.add_argument("--threshold", "-t", type=float, default=50.0)

    p_glr = subs.add_parser("add_glare", help="Glare blend")
    p_glr.add_argument("--strength", type=float, default=0.85)

    p_wc = subs.add_parser("add_watercolor_texture", help="Watercolor")
    p_wc.add_argument("--intensity", type=float, default=0.3)

    return parser.parse_args()


def main():
    args = cli_arg_parser()

    if args.op == "about":
        print(
            "Applies image operations: resize, sepia, vignette, pixelate, frames, glare, watercolor.\n"
            "Usage: python main.py --image in.jpg OP [flags]\n"
            "Notes: resize needs --height and --width; pixelate coords are (y,x); add_rect_frame uses B,G,R.\n"
        )
        return

    if not args.image:
        print("Error: --image is required.", file=sys.stderr)
        sys.exit(2)

    img = cv.imread(args.image, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)

    if args.op == "resize":
        out_img = resize(img, args.height, args.width, args.sigma)
    elif args.op == "sepia_filter":
        out_img = sepia_filter(img, args.intensity)
    elif args.op == "vignette":
        out_img = vignette(img, args.intensity, args.radius)
    elif args.op == "pixelate":
        out_img = pixelate(img, (args.y1, args.x1), (args.y2, args.x2), args.block)
    elif args.op == "add_rect_frame":
        out_img = add_rect_frame(img, args.thickness, (args.b, args.g, args.r))
    elif args.op == "add_figure_frame":
        out_img = add_figure_frame(img, args.index, args.threshold)
    elif args.op == "add_glare":
        out_img = add_glare(img, args.strength)
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