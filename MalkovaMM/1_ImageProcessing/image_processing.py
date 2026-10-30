import abc
import numpy as np
import cv2


class ImageFilter(abc.ABC):
    def __init__(self, image, params=None):
        self.image = image
        self.params = params

    @staticmethod
    def get_filter(image, filter_type):
        if filter_type == 'resize':
            return Resize(image)
        elif filter_type == 'grayscale':
            return Grayscale(image)
        elif filter_type == 'antique':
            return Antique(image)
        elif filter_type == 'fadecolor':
            return FadeColor(image)
        elif filter_type == 'imitfilm':
            return ImitFilm(image)
        elif filter_type == 'matte':
            return Matte(image)
        elif filter_type == 'noisescratches':
            return NoiseScratches(image)
        elif filter_type == 'neonlight':
            return NeonLight(image)
        else:
            raise ValueError(f'Фильтра {filter_type} не существует')

    @abc.abstractmethod
    def apply_filter(self):
        raise ValueError(f'Not implemented')


class Resize(ImageFilter):
    def __init__(self, image, params=None):
        super().__init__(image, params)

    def apply_filter(self):
        h, w, colors = self.image.shape
        nh = h - (h%2)
        nw = w - (w%2)
        nimage = self.image[:nh, :nw]
        reshaped = nimage.reshape(h//2, 2, w//2, 2, colors)
        result = reshaped.mean(axis=(1,3))
        return np.clip(result, 0, 255).astype(np.uint8)


class Grayscale(ImageFilter):
    def __init__(self, image, params=None):
        super().__init__(image, params)

    def apply_filter(self):
        b = self.image[:,:,0].astype(np.float32)
        g = self.image[:,:,1].astype(np.float32)
        r = self.image[:,:,2].astype(np.float32)
        gray = 0.144*b + 0.587*g + 0.299*r
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        result = np.stack((gray, gray, gray), axis=2)
        return result


class Antique(ImageFilter):
    def __init__(self, image, params=None):
        super().__init__(image, params)

    def apply_filter(self):
        texture = cv2.imread('images/texture.jpg')
        if texture is None:
            raise ValueError("Не удалось отфильтровать фотографию")
        h, w = self.image.shape[:2]
        texture = cv2.resize(texture, (w,h))
        img1 = self.image.astype(np.float32)/255.0
        tex1 = texture.astype(np.float32)/255.0
        result = img1*tex1
        result = result*255.0
        noise = np.random.normal(0,10, self.image.shape)
        result = result+noise
        return np.clip(result, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    def __init__(self, image, params=None):
        super().__init__(image, params)

    def apply_filter(self):
        b = self.image[:,:,0].astype(np.float32)
        g = self.image[:,:,1].astype(np.float32)
        r = self.image[:,:,2].astype(np.float32)
        nr = r*0.6 + 50
        ng = g*0.6 + 50
        nb = b*0.6 + 50
        result = np.stack((nb, ng, nr), axis=2)
        return np.clip(result, 0, 255).astype(np.uint8)


class ImitFilm(ImageFilter):
    def __init__(self, image, params=None):
        super().__init__(image, params)

    def apply_filter(self):
        b = self.image[:,:,0].astype(np.float32)
        g = self.image[:,:,1].astype(np.float32)
        r = self.image[:,:,2].astype(np.float32)
        nr = r*0.7
        ng = g*0.6
        nb = b*0.4
        noise = np.random.normal(0,10, self.image.shape)
        result = np.stack((nb, ng, nr), axis=2)
        result = (result + noise)*0.9
        return np.clip(result, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    def __init__(self, image, params=None):
        super().__init__(image, params)

    def apply_filter(self):
        h, w, colors = self.image.shape
        cx, cy = w/2, h/2
        Y, X = np.ogrid[:h, :w]
        dist = ((X-cx)/(cx*0.8))**2 + ((Y-cy)/(cy*0.7))**2
        mask = np.clip(1.5-dist, 0, 1)
        img_float = self.image.astype(np.float32)
        result = img_float*mask[:, :, np.newaxis]
        white = 255*(1-mask[:, :, np.newaxis])
        result = result + white
        return np.clip(result, 0, 255).astype(np.uint8)



class NoiseScratches(ImageFilter):
    def __init__(self, image, params=None):
        super().__init__(image, params)

    def apply_filter(self):
        img_float = self.image.astype(np.float32)
        h, w, colors = self.image.shape
        noise = np.random.normal(0,25, img_float.shape)
        img_noise = img_float + noise
        num_scrt = 10
        scrt_x = np.random.randint(0, w, num_scrt)
        for x in scrt_x:
            start_y = np.random.randint(0, h//2)
            end_y = np.random.randint(h//2, h)
            img_noise[start_y: end_y, x, :] += 60
        return np.clip(img_noise, 0, 255).astype(np.uint8)


class NeonLight(ImageFilter):
    def __init__(self, image, params=None):
        super().__init__(image, params)

    def apply_filter(self):
        b = self.image[:,:,0].astype(np.float32)
        g = self.image[:,:,1].astype(np.float32)
        r = self.image[:,:,2].astype(np.float32)
        gray = 0.114*b + 0.587*g + 0.299*r
        blurred = cv2.blur(gray, (3,3))
        gy = np.array([[-1,-2,-1],
                       [0,0,0],
                       [1,2,1]], dtype=np.float32)
        gx = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]], dtype=np.float32)
        Gx = cv2.filter2D(blurred,cv2.CV_32F,gx)
        Gy = cv2.filter2D(blurred,cv2.CV_32F,gy)
        magnitude = np.abs(Gx) + np.abs(Gy)
        edges = np.clip(magnitude, 0, 255)
        neonlines = np.zeros_like(self.image,dtype=np.float32)
        neonlines[:,:, 2] = edges*0.5
        neonlines[:,:, 0] = edges
        dark_image = self.image.astype(np.float32)*0.3
        result = dark_image + neonlines
        return np.clip(result, 0, 255).astype(np.uint8)

