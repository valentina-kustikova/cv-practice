import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog, simpledialog, Canvas
from PIL import Image, ImageTk


def Print_Image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return 0
    
def Change_Res(img, nx, ny):
    h, w, c = img.shape

    x_idx = (np.linspace(0, w - 1, nx)).astype(int)
    y_idx = (np.linspace(0, h - 1, ny)).astype(int)

    result = img[y_idx[:, None], x_idx[None, :], :]
    return result.astype(np.uint8)


def Sep(img):
    img = img.astype(np.float32)
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    tr = 0.393 * r + 0.769 * g + 0.189 * b
    tg = 0.349 * r + 0.686 * g + 0.168 * b
    tb = 0.272 * r + 0.534 * g + 0.131 * b
    sepia = np.stack([tb, tg, tr], axis=2)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


def Vinyetka(img):
    h, w, c = img.shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    diag = np.sqrt(cx**2 + cy**2)
    coef = 1 - (dist / diag)**2
    coef = np.clip(coef, 0, 1)
    vinetka = img * coef[..., np.newaxis]
    return vinetka.astype(np.uint8)

def Take_Points(image):
    points = []

    def Click_Event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.imshow('image', image)
    cv2.setMouseCallback('image', Click_Event)

    while True:
        cv2.waitKey(1)
        if len(points) == 2:
            break

    cv2.destroyAllWindows()
    
    return(points)

def Pixels(image, points):
    h, w, c = image.shape
    block_size = 20
    x_l, x_r = sorted([points[0][0], points[1][0]])
    y_l, y_r = sorted([points[0][1], points[1][1]])

    img_copy = image.copy()
    region = img_copy[y_l:y_r, x_l:x_r]

    ry, rx, _ = region.shape

    ny = ry // block_size
    nx = rx // block_size
    trimmed = region[:ny * block_size, :nx * block_size]


    small = trimmed.reshape(ny, block_size, nx, block_size, 3).mean(axis=(1, 3), dtype=int)
    pixelated = np.repeat(np.repeat(small, block_size, axis=0), block_size, axis=1)

    img_copy[y_l:y_l + pixelated.shape[0], x_l:x_l + pixelated.shape[1]] = pixelated

    return img_copy


def Rec_Frame(image, thick):
    h, w, c = image.shape
    img = image.copy()
    color = np.array([120, 50, 200], dtype=np.uint8)
    img[:, :thick] = color
    img[:, w-thick:] = color
    img[:thick, :] = color
    img[h-thick:, :] = color
    return img


def Frames(image, n_frame):
    h, w, c = image.shape
    path_names = ["Frame1.png", "Frame2.png", "Frame3.png"]
    frame = cv2.imread(path_names[n_frame])
    frame = Change_Res(frame, w, h)
    
    mask = ~(frame == [255, 255, 255]).all(axis=2)
    image[mask] = frame[mask]
    return image


def Camera_Light(image):
    h, w, c = image.shape
    light = cv2.imread("blind.png")
    light = Change_Res(light, w // 2, h // 2)
    mask = ~(light == [255, 255, 255]).all(axis=2)
    overlay = (0.8 * light + 0.2 * image[:h//2, :w//2]).astype(np.uint8)
    image[:h//2, :w//2][mask] = overlay[mask]
    return image


def Aqua_Paper(image):
    h, w, c = image.shape
    aqua = cv2.imread("aqua.png")
    aqua = Change_Res(aqua, w, h)
    result = (0.3 * aqua + 0.7 * image).astype(np.uint8)
    return result

##########################Графический интерфейс##############################################

class ImageEditor:
    def __init__(self, root):
        self.MAX_WIDTH = 800
        self.MAX_HEIGHT = 600
        self.root = root
        self.root.title("Image Editor")
        
        self.img_original = None
        self.img_current = None
        self.tk_img = None
        
        # Элементы интерфейса
        self.label = Label(root)
        self.label.pack()
        
        btn_frame = [
            ("Добавтить изображение", self.open_image),
            ("Функция изменения разрешения изображения.", self.change_resolution),
            ("Функция применения фотоэффекта сепии к изображению.", self.apply_sepia),
            ("Функция применения фотоэффекта виньетки к изображению.", self.apply_vignette),
            ("Функция пикселизации заданной прямоугольной области изображения.", self.apply_pixels),
            ("Функция наложения прямоугольной одноцветной рамки заданной ширины по краям изображения.", self.apply_frame),
            ("Функция наложения фигурной одноцветной рамки по краям изображения. Тип фигурной рамки является параметром функции.", self.apply_custom_frame),
            ("Функция наложения эффекта бликов объектива камеры", self.apply_camera_light),
            ("Функция наложения текстуры акварельной бумаги", self.apply_aqua),
            ("Сброс", self.reset_image),
            ("Сохранить", self.save_image),
        ]
        
        for text, cmd in btn_frame:
            Button(root, text=text, command=cmd).pack(fill='x')
    
    def open_image(self):
        path = filedialog.askopenfilename(title="Выберите изображение",
                                          filetypes=[("Изображения", "*.png *.jpg")])
        if not path:
            return

        with open(path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            self.img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if self.img_original is None:
            print("Не удалось открыть изображение!")
            return
        
        self.img_current = self.img_original.copy()
        self.show_image()
    
    def show_image(self):
        img_rgb = cv2.cvtColor(self.img_current, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        w, h = img_pil.size
        scale = min(self.MAX_WIDTH / w, self.MAX_HEIGHT / h, 1)
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.label.config(image=self.tk_img)
        self.label.image = self.tk_img
        
    def change_resolution(self):
        if self.img_current is None:
            return
        
        nx = simpledialog.askinteger("Изменение разрешения", "Введите новую ширину:")
        ny = simpledialog.askinteger("Изменение разрешения", "Введите новую высоту:")
        
        if nx is not None and ny is not None:
            self.img_current = Change_Res(self.img_current, nx, ny)
            self.show_image()
    
    def apply_sepia(self):
        self.img_current = Sep(self.img_current)
        self.show_image()
    
    def apply_vignette(self):
        self.img_current = Vinyetka(self.img_current)
        self.show_image()
    
    def apply_pixels(self):
        print("Выберите область для пикселизации в отдельном окне...")
        points = Take_Points(self.img_current.copy())
        self.img_current = Pixels(self.img_current, points)
        self.show_image()
    
    def apply_frame(self):
        thick = simpledialog.askinteger("Рамка", "Введите толщину рамки:")
        if thick is not None:
            self.img_current = Rec_Frame(self.img_current, thick)
            self.show_image()
    
    def apply_custom_frame(self):
        n_frame = simpledialog.askinteger("Кружевная рамка", "Выберите номер рамки (0-2):")
        if n_frame is not None:
            self.img_current = Frames(self.img_current, n_frame)
            self.show_image()
    
    def apply_camera_light(self):
        self.img_current = Camera_Light(self.img_current)
        self.show_image()
    
    def apply_aqua(self):
        self.img_current = Aqua_Paper(self.img_current)
        self.show_image()
    
    def reset_image(self):
        if self.img_original is not None:
            self.img_current = self.img_original.copy()
            self.show_image()
    
    def save_image(self):
        if self.img_current is None:
            return
        path = filedialog.asksaveasfilename(title="Сохранить изображение",
                                            defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self.img_current)
            print(f"Изображение сохранено: {path}")


if __name__ == "__main__":
    root = Tk()
    app = ImageEditor(root)
    root.mainloop()

