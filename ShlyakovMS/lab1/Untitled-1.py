import cv2
import numpy as np
import math
from tkinter import Tk, Button, Label, filedialog, simpledialog, Canvas
from PIL import Image, ImageTk


def Print_Image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return 0
    
def Change_Res(img, nx, ny):
    result = np.zeros((ny,nx,3), np.uint8)
    h, w, c = img.shape
    for x in range(nx):
        for y in range(ny):
            result[y][x] = img[y*h//ny][x*w//nx]
        
    return result

def Sep(img):
    h, w, c = img.shape
    for x in range(w):
        for y in range(h):
            b, g, r = img[y][x]
            r = min(r * 0.393 + g * 0.769 + b * 0.189, 255)
            g = min(r * 0.349 + g * 0.686 + b * 0.168, 255)
            b = min(r * 0.272 + g * 0.534 + b * 0.131, 255)
            img[y][x] = b, g, r
    return img

def Vinyetka(img):
    h, w, c = img.shape
    centr = (w//2, h//2)
    diag = math.sqrt(math.pow(centr[0], 2) + math.pow(centr[1], 2))
    for x in range(w):
        for y in range(h):
            dist = math.sqrt((math.pow(x - centr[0], 2) + math.pow(y - centr[1], 2)))
            if dist > 0:
                b, g, r = img[y][x]
                coef = 1 - math.pow(dist/diag, 2)
                r = r * coef
                g = g * coef
                b = b * coef
                img[y][x] = b, g, r
    return img

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
    
    for x in range(x_l, x_r, block_size):
        for y in range(y_l, y_r, block_size):
            
            x_border = min(x + block_size, x_r)
            y_border = min(y + block_size, y_r)
            
            image_part = image[y:y_border, x:x_border]
            
            mean_color = image_part.mean(axis = (0,1), dtype = int)
            
            image[y:y_border, x:x_border] = mean_color
    return(image)

def Rec_Frame(image, thick):
    h, w, c = image.shape
    image[0:h, 0:thick] = [120, 50, 200]
    image[0:h, w-thick:w] = [120, 50, 200]
    image[0:thick, 0:w] = [120, 50, 200]
    image[h-thick:h, 0:w] = [120, 50, 200]
    return image

def Frames(image, n_frame):
    h, w, c = image.shape
    path_names = ["Frame1.png", "Frame2.png", "Frame3.png"]
    frame = cv2.imread(path_names[n_frame])
    
    frame = Change_Res(frame, w, h)
    
    for x in range(w):
        for y in range(h):
            if(not np.all(frame[y][x]==[255,255,255])):
                image[y][x] = frame[y][x]
    
    return image
    
def Camera_Light(image):
    h, w, c = image.shape
    light = cv2.imread("blind.png")
    light = Change_Res(light, w//2, h//2)
    
    for x in range(w//2):
        for y in range(h//2):
            if(not np.all(light[y][x]==[255,255,255])):
                image[y][x] = (0.8 * light[y][x] + 0.2 * image[y][x]).astype(np.uint8)
    return image

def Aqua_Paper(image):
    h, w, c = image.shape
    aqua = cv2.imread("aqua.png")
    aqua = Change_Res(aqua, w, h)
    
    for x in range(w):
        for y in range(h):
            image[y][x] = (0.3 * aqua[y][x] + 0.7 * image[y][x]).astype(np.uint8)
    return image

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

