import cv2
import numpy as np
import argparse
import sys

def parse_color(s):
    try:
        b, g, r = map(int, s.split(','))
        return (b, g, r)
    except Exception:
        raise argparse.ArgumentTypeError("Color must be in format 'R,G,B' (e.g. '0,0,255')")

def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',
                        help="Mode ('image', 'sepia', 'resize', 'vignette', 'pixelate_area', "
                             "'add_border', 'frame_from_image', 'flares_from_image', 'texture_blend')",
                        type=str,
                        dest='mode',
                        default='image')

    parser.add_argument('--source',
                        help='Path to the source/overlay image (used for texture, flares, or frame)',
                        type=str,
                        dest='source_image_path',
                        default=None)

    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='test_out.jpg',
                        dest='out_image_path')
    parser.add_argument('--width',
                        help='New width (in resize mode), area width (in pixelate_area mode), or border width (in add_border mode)',
                        type=int,
                        default=-1,
                        dest='width')
    parser.add_argument('--height',
                        help='New height (in resize mode) or area height (in pixelate_area mode)',
                        type=int,
                        default=-1,
                        dest='height')

    parser.add_argument('--color',
                        help='Color of border in BGR format',
                        type=parse_color,
                        default=(0, 0, 255),
                        dest='color')

    parser.add_argument('--strength',
                        help='Strength of the effect (e.g., for vignette)',
                        type=float,
                        default=0.8,
                        dest='strength')
    parser.add_argument('--softness',
                        help='Softness of the effect (e.g., for vignette)',
                        type=float,
                        default=0.4,
                        dest='softness')
    parser.add_argument('--x',
                        help='First coordinate of pixelate area',
                        type=int,
                        default=0,
                        dest='x')
    parser.add_argument('--y',
                        help='Second coordinate of pixelate area',
                        type=int,
                        default=0,
                        dest='y')
    parser.add_argument('--block_size',
                        help='Size of pixelate block',
                        type=int,
                        default=15,
                        dest='block_size')

    parser.add_argument('--threshold',
                        help='Brightness threshold (0-255) for flare extraction',
                        type=int,
                        default=220,
                        dest='threshold')
    parser.add_argument('--alpha',
                        help='Blending alpha (0.0-1.0) for texture overlay',
                        type=float,
                        default=0.3,
                        dest='alpha')
    parser.add_argument('--mask_threshold',
                        help='mask_threshold',
                        type=int,
                        default=1,
                        dest='mask_threshold')
    args = parser.parse_args()
    return args

def mode_sepia(image, alpha=1.0):
    img = image.copy().astype(np.float32)
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia_img = img @ sepia_matrix.T
    sepia_img = np.clip(sepia_img, 0, 255)
    final_img = (1.0 - alpha) * img + alpha * sepia_img
    return final_img.astype(np.uint8)


def mode_resize(image, new_width, new_height):

    original_height, original_width = image.shape[:2]

    x_new = np.arange(new_width)
    y_new = np.arange(new_height)

    alpha_w = (original_width / new_width)
    alpha_h = (original_height / new_height)
    x_old = (x_new * alpha_w).astype(np.int32)
    y_old = (y_new * alpha_h).astype(np.int32)

    xx, yy = np.meshgrid(x_old, y_old)
    resized_image = image[yy, xx]

    return resized_image


def mode_vignette(image, strength=0.8, softness=0.4):

    height, width = image.shape[:2]

    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)

    xx, yy = np.meshgrid(x, y)

    radius = np.sqrt(xx ** 2 + yy ** 2)

    mask = radius / (softness + 1e-6)

    mask = (1.0 - strength) + strength * (1.0 - np.clip(mask, 0, 1))

    if image.ndim == 3:
        mask = mask[..., np.newaxis]

    vignette_image = (image.astype(np.float32) * mask).astype(np.uint8)

    return vignette_image

def pixelate_area(image, x, y, width, height, block_size=15):
    if block_size < 0:
        raise ValueError("Block size must be not negative")

    img = image.copy()

    x_end = min(x + width, img.shape[1])
    y_end = min(y + height, img.shape[0])
    x_start = max(x, 0)
    y_start = max(y, 0)

    roi = img[y_start:y_end, x_start:x_end]

    (h, w) = roi.shape[:2]

    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            r_end_block = min(r + block_size, h)
            c_end_block = min(c + block_size, w)

            block = roi[r:r_end_block, c:c_end_block]

            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1), dtype=int)

                roi[r:r_end_block, c:c_end_block] = avg_color

    return img


def add_border(image, border_width, color=(0, 0, 255)):

    img_with_border = image.copy()

    h, w = img_with_border.shape[:2]

    img_with_border[0:border_width, :] = color

    img_with_border[h - border_width:h, :] = color

    img_with_border[:, 0:border_width] = color

    img_with_border[:, w - border_width:w] = color

    return img_with_border

def rgb2gray_numpy(image_bgr):
    if image_bgr.ndim == 2:
        return image_bgr
    return np.dot(image_bgr[...,:3], [0.114, 0.587, 0.299])

def apply_frame_from_image(target_image, frame_source_image, thickness=100, mask_threshold=20, alpha=1.0):

    h_t, w_t = target_image.shape[:2]
    h_f, w_f = frame_source_image.shape[:2]

    if h_f < 2 * thickness or w_f < 2 * thickness:
        raise ValueError("Изображение рамки слишком маленькое для заданной толщины.")

    tl, tr = frame_source_image[:thickness, :thickness], frame_source_image[:thickness, w_f-thickness:]
    bl, br = frame_source_image[h_f-thickness:, :thickness], frame_source_image[h_f-thickness:, w_f-thickness:]
    top = frame_source_image[:thickness, thickness:w_f-thickness]
    bottom = frame_source_image[h_f-thickness:, thickness:w_f-thickness]
    left = frame_source_image[thickness:h_f-thickness, :thickness]
    right = frame_source_image[thickness:h_f-thickness, w_f-thickness:]

    tw, th = w_t - 2*thickness, h_t - 2*thickness
    top, bottom = mode_resize(top, tw, thickness), mode_resize(bottom, tw, thickness)
    left, right = mode_resize(left, thickness, th), mode_resize(right, thickness, th)

    result = target_image.copy().astype(np.float32)

    def create_mask(part):
        if part.shape[2] == 4:
            return part[..., 3] > mask_threshold
        else:
            gray = rgb2gray_numpy(part)
            return gray > mask_threshold

    def blend(roi, overlay, mask):
        m = mask[..., np.newaxis].astype(np.float32)
        return roi*(1 - alpha*m) + overlay[..., :3]*alpha*m

    def apply(x, y, part):
        h, w = part.shape[:2]
        roi = result[y:y+h, x:x+w]
        mask = create_mask(part)
        result[y:y+h, x:x+w] = blend(roi, part, mask)

    apply(0, 0, tl)
    apply(w_t-thickness, 0, tr)
    apply(0, h_t-thickness, bl)
    apply(w_t-thickness, h_t-thickness, br)
    apply(thickness, 0, top)
    apply(thickness, h_t-thickness, bottom)
    apply(0, thickness, left)
    apply(w_t-thickness, thickness, right)

    return np.clip(result, 0, 255).astype(np.uint8)

def apply_flares_from_image(target_image, flare_source_image, threshold=220, alpha=1):

    h_t, w_t = target_image.shape[:2]
    h_f, w_f = flare_source_image.shape[:2]
    if (h_t, w_t) != (h_f, w_f):
        flare_source_image = mode_resize(flare_source_image, w_t, h_t)

    luminance = np.mean(flare_source_image.astype(np.float32), axis=2)

    mask = np.clip((luminance - threshold) / (255 - threshold), 0, 1)
    flares = flare_source_image.astype(np.float32) * mask[..., np.newaxis]

    result = target_image.astype(np.float32) + alpha * flares
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_texture_blend(target_image, texture_image, alpha=0.3):

    h_t, w_t = target_image.shape[:2]
    h_f, w_f = texture_image.shape[:2]

    resized_texture = texture_image
    if (h_t, w_t) != (h_f, w_f):
        resized_texture = mode_resize(texture_image, w_t, h_t)

    target_float = target_image.astype(np.float32)
    texture_float = resized_texture.astype(np.float32)

    blended_image = (1.0 - alpha) * target_float + alpha * texture_float
    return np.clip(blended_image, 0, 255).astype(np.uint8)


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at the path: {image_path}")
    return img

def process_image(img, args):
    if args.mode not in ['resize', 'add_border', 'pixelate_area'] and args.width != -1:
        raise ValueError('Unknown argument \'width\'')
    if args.mode not in ['resize', 'pixelate_area'] and args.height != -1:
        raise ValueError('Unknown argument \'height\'')
    if args.mode == 'image':
        return img
    elif args.mode == 'sepia':
        return mode_sepia(img)
    elif args.mode == 'resize':
        if args.width <= 0 or args.height <= 0:
            raise ValueError('Unspecified or invalid target width and height')
        return mode_resize(img, args.width, args.height)
    elif args.mode == 'vignette':
        return mode_vignette(img, args.strength, args.softness)
    elif args.mode == 'pixelate_area':
        if args.width <= 0 or args.height <= 0:
            raise ValueError('Invalid area size for pixelation')
        if args.block_size <= 0:
            raise ValueError('Invalid block size for pixelation')
        return pixelate_area(img, args.x, args.y, args.width, args.height, args.block_size)
    elif args.mode == 'add_border':
        if args.width <= 0:
            raise ValueError('Unspecified or invalid border width')
        return add_border(img, args.width, args.color)
    elif args.mode == 'frame_from_image':
        if not args.source_image_path:
            raise ValueError("Mode 'frame_from_image' requires --source argument")
        source_img = load_image(args.source_image_path)
        return apply_frame_from_image(img, source_img, args.width, args.mask_threshold, args.alpha)
    elif args.mode == 'flares_from_image':
        if not args.source_image_path:
            raise ValueError("Mode 'flares_from_image' requires --source argument")
        source_img = load_image(args.source_image_path)
        return apply_flares_from_image(img, source_img, args.threshold, args.alpha)
    elif args.mode == 'texture_blend':
        if not args.source_image_path:
            raise ValueError("Mode 'texture_blend' requires --source argument")
        source_img = load_image(args.source_image_path)
        return apply_texture_blend(img, source_img, args.alpha)
    else:
        raise ValueError('Unsupported mode')

def save_image(img, output_path):
    cv2.imwrite(output_path, img)

def display(img, processed):
    cv2.imshow('Original image', img)
    cv2.imshow('Processed image', processed)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    args = cli_argument_parser()
    try:
        img = load_image(args.image_path)
        processed = process_image(img, args)
        save_image(processed, args.out_image_path)
        display(img, processed)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    sys.exit(main() or 0)