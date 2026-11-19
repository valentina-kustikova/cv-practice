import abc
import numpy as np
import cv2


def adapt_texture(texture: np.ndarray, target_h: int, target_w: int) -> np.ndarray:

    th, tw = texture.shape[:2]

    target_is_landscape = target_w > target_h
    texture_is_landscape = tw > th

    if target_is_landscape != texture_is_landscape:
        texture = np.rot90(texture)
        th, tw = texture.shape[:2]

    x_indices = np.clip((np.arange(target_w) * (tw / target_w)).astype(int), 0, tw - 1)
    y_indices = np.clip((np.arange(target_h) * (th / target_h)).astype(int), 0, th - 1)

    return texture[y_indices[:, None], x_indices]


class ImageFilter(abc.ABC):
    def __init__(self, image, params):
        self.image = image
        self.params = params

    @staticmethod
    def get_filter(image, params):
        if params['filter'] == "grayscale": 
            return Grayscale(image, params)
        elif params['filter'] == "resize": 
            return Resize(image, params)
        elif params['filter'] == "sepia": 
            return Sepia(image, params)
        elif params['filter'] == "fadecolor": 
            return FadeColor(image, params)
        elif params['filter'] == "filmsimulation": 
            return FilmSimulation(image, params)
        elif params['filter'] == "matte": 
            return Matte(image, params)
        elif params['filter'] == "oldphoto": 
            return OldPhotoNoise(image, params)
        elif params['filter'] == "neon": 
            return NeonEffect(image, params)
        else:
            raise ValueError(f"No filter: {params['filter']}")

    @abc.abstractmethod
    def apply_filter(self):
        raise ValueError(f'Not implemented')


class Resize(ImageFilter):
    def __init__(self, image, params):
        super().__init__(image, params)

    def apply_filter(self):
        scale = float(self.params.get('scale', 1.0))
        if scale <= 0:
            raise ValueError("'scale' must be > 0")
        h, w = self.image.shape[:2]

        new_h = int(h * scale)
        new_w = int(w * scale)

        x_indices = np.clip((np.arange(new_w) / scale).astype(int), 0, w - 1) 
        y_indices = np.clip((np.arange(new_h) / scale).astype(int), 0, h - 1)

        return self.image[y_indices[:, None], x_indices]


class Grayscale(ImageFilter):
    def __init__(self, image, params):
        super().__init__(image, params)
    
    def apply_filter(self):
        blue = self.image [:,:,0]
        green = self.image [:,:,1]
        red = self.image [:,:,2]

        gray = 0.114*blue + 0.587*green + 0.299*red
        gray_uint8 = gray.astype(np.uint8)

        return gray_uint8 


class Sepia(ImageFilter):
    def __init__(self, image, params):
        super().__init__(image, params)

    def apply_filter(self):
        blue = self.image [:,:,0]
        green = self.image [:,:,1]
        red = self.image [:,:,2]

        new_blue = 0.131*blue + 0.534*green + 0.272*red
        new_green = 0.168*blue + 0.686*green + 0.349*red
        new_red = 0.189*blue + 0.769*green + 0.393*red

        sepia = np.stack((new_blue, new_green, new_red), axis = -1) 

        return np.clip(sepia, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    def __init__(self, image, params):
        super().__init__(image, params)

    def apply_filter(self):
        the_power_of_fading = float(self.params.get('the_power_of_fading', 0.25))
        fade_color = 210.0 
        img_float = self.image.astype(np.float32)
        img_faded = img_float * (1 - the_power_of_fading) + fade_color * the_power_of_fading

        return np.clip(img_faded, 0, 255).astype(np.uint8)


class FilmSimulation(ImageFilter):
    def __init__(self, image, params):
        super().__init__(image, params)

    def apply_filter(self):
        blue = self.image [:,:,0].astype(np.float32)
        green = self.image [:,:,1].astype(np.float32)
        red = self.image [:,:,2].astype(np.float32)

        i_channel = 1.2 * red + 0.6 * green - 0.5 * blue
        i_img = np.clip(i_channel, 0, 255).astype(np.uint8)
        i_img_3d = np.stack((i_img, i_img, i_img), axis=-1)

        texture_path = self.params.get('texture', 'textures/grain_texture.jpg')
        texture = cv2.imread(texture_path)

        if texture is None:
            raise ValueError(f"No texture")

        h, w = self.image.shape[:2]
        fitted_texture = adapt_texture(texture, h, w)
        blended = i_img_3d.astype(np.float32) * 0.8 + fitted_texture.astype(np.float32) * 0.2

        return np.clip(blended, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    def __init__(self, image, params):
        super().__init__(image, params)

    def apply_filter(self):
        h, w = self.image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        y, x = np.ogrid[:h, :w]
        
        d = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        oval_min_r = float(self.params.get('oval_min_r', 0.65))
        oval_max_r = float(self.params.get('oval_max_r', 1.0))

        k = (d - oval_min_r) / (oval_max_r - oval_min_r)
        k = np.clip(k, 0.0, 1.0)
        k_3d = k[:, :, None]

        white = np.full_like(self.image, 255, dtype=np.float32)
        result = self.image.astype(np.float32) * (1.0 - k_3d) + white * k_3d

        return np.clip(result, 0, 255).astype(np.uint8)


class OldPhotoNoise(ImageFilter):
    def __init__(self, image, params):
        super().__init__(image, params)

    def apply_filter(self):
        texture_path = self.params.get('texture', 'textures/old_paper.jpg')
        texture = cv2.imread(texture_path)
        if texture is None:
            raise ValueError(f"No texture")

        h, w = self.image.shape[:2]
        fitted_texture = adapt_texture(texture, h, w)

        img_float = self.image.astype(np.float32)
        tex_float = fitted_texture.astype(np.float32)
        result = (img_float * tex_float) / 255.0
        
        return np.clip(result, 0, 255).astype(np.uint8)


class NeonEffect(ImageFilter):
    def __init__(self, image, params):
        super().__init__(image, params)

    def apply_filter(self):
        blue = self.image [:,:,0]
        green = self.image [:,:,1]
        red = self.image [:,:,2]

        gray_float = (0.114 * blue + 0.587 * green + 0.299 * red).astype(np.float32)

        blur = cv2.GaussianBlur(gray_float, (5, 5), 0)

        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)

        ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

        gx = cv2.filter2D(blur, cv2.CV_32F, kx)
        gy = cv2.filter2D(blur, cv2.CV_32F, ky)

        magn = np.sqrt(gx ** 2 + gy ** 2)
        threshold = float(self.params.get('threshold', 50.0))

        edges = np.where(magn > threshold, magn, 0.0)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        glow = cv2.GaussianBlur(edges, (7, 7), 0)

        result = np.zeros_like(self.image)
        result[:, :, 0] = glow 
        result[:, :, 1] = glow
        result[:, :, 2] = (glow * 0.2).astype(np.uint8)

        return result