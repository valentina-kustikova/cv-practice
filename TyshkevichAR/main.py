import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import math
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'filters'))

from filters.resolution import change_resolution
from filters.sepia import apply_sepia
from filters.vignette import apply_vignette
from filters.pixelation import apply_pixelation
from filters.simple_border import apply_simple_border
from filters.shaped_border import apply_shaped_border
from filters.lens_flare import apply_lens_flare
from filters.watercolor_paper import apply_watercolor_paper


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Редактор")
        self.root.geometry("1200x700")
        self.original_image = None
        self.processed_image = None
        self.current_image = None
        self.setup_ui()

    def setup_ui(self):

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.LabelFrame(main_frame, text="Фильтры", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.load_btn = ttk.Button(control_frame, text="Загрузить изображение",
                                   command=self.load_image)
        self.load_btn.pack(fill=tk.X, pady=5)

        self.clear_btn = ttk.Button(control_frame, text="Очистить фильтры",
                                    command=self.clear_filters, state=tk.DISABLED)
        self.clear_btn.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Выберите фильтр:").pack(anchor=tk.W, pady=(10, 5))

        self.filter_var = tk.StringVar()
        filters = [
            "Исходное изображение",
            "Изменение разрешения",
            "Сепия",
            "Виньетка",
            "Пиксели",
            "Простая рамка",
            "Фигурная рамка",
            "Блик объектива",
            "Акварельная бумага"
        ]

        self.filter_combo = ttk.Combobox(control_frame, textvariable=self.filter_var,
                                         values=filters, state="readonly")
        self.filter_combo.set("Исходное изображение")
        self.filter_combo.pack(fill=tk.X, pady=5)
        self.filter_combo.bind('<<ComboboxSelected>>', self.apply_selected_filter)

        self.param_frame = ttk.Frame(control_frame)
        self.param_frame.pack(fill=tk.X, pady=10)

        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        orig_frame = ttk.LabelFrame(image_frame, text="Исходное изображение", padding=5)
        orig_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5))

        self.original_canvas = tk.Canvas(orig_frame, bg='beige', width=400, height=400)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        proc_frame = ttk.LabelFrame(image_frame, text="Обработанное изображение", padding=5)
        proc_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0))

        self.processed_canvas = tk.Canvas(proc_frame, bg='beige', width=400, height=400)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png")
            ]
        )

        if file_path:
            try:
                # Используем PIL для загрузки
                pil_image = Image.open(file_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')

                image_array = np.array(pil_image)

                # Конвертируем из RGB в BGR для OpenCV
                self.original_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                # И обратно в RGB для отображения
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

                self.current_image = self.original_image.copy()

                self.clear_btn.config(state=tk.NORMAL)
                self.display_images()

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")

    def clear_filters(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.filter_combo.set("Исходное изображение")
            self.clear_param_frame()
            self.display_images()

    def apply_selected_filter(self, event=None):
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        filter_name = self.filter_var.get()

        try:
            if filter_name == "Исходное изображение":
                self.current_image = self.original_image.copy()
                self.clear_param_frame()

            elif filter_name == "Изменение разрешения":
                self.show_resolution_params()
                return

            elif filter_name == "Сепия":
                self.current_image = apply_sepia(self.original_image.copy())
                self.clear_param_frame()

            elif filter_name == "Виньетка":
                self.current_image = apply_vignette(self.original_image.copy())
                self.clear_param_frame()

            elif filter_name == "Пиксели":
                self.show_pixelation_params()
                return

            elif filter_name == "Простая рамка":
                self.show_simple_border_params()
                return

            elif filter_name == "Фигурная рамка":
                self.show_shaped_border_params()
                return

            elif filter_name == "Блики объектива":
                self.current_image = apply_lens_flare(self.original_image.copy())
                self.clear_param_frame()

            elif filter_name == "Акварельная бумага":
                self.current_image = apply_watercolor_paper(self.original_image.copy())
                self.clear_param_frame()

            self.display_images()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка применения фильтра:\n{str(e)}")

    def show_resolution_params(self):
        self.clear_param_frame()

        ttk.Label(self.param_frame, text="Новая ширина:").pack(anchor=tk.W)
        width_var = tk.IntVar(value=min(800, self.original_image.shape[1]))
        width_entry = ttk.Entry(self.param_frame, textvariable=width_var)
        width_entry.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(self.param_frame, text="Новая высота:").pack(anchor=tk.W)
        height_var = tk.IntVar(value=min(600, self.original_image.shape[0]))
        height_entry = ttk.Entry(self.param_frame, textvariable=height_var)
        height_entry.pack(fill=tk.X, pady=(0, 10))

        def apply_res():
            try:
                width = width_var.get()
                height = height_var.get()
                if width <= 0 or height <= 0:
                    messagebox.showerror("Ошибка", "Ширина и высота должны быть положительными числами")
                    return

                self.current_image = change_resolution(
                    self.original_image.copy(),
                    width,
                    height
                )
                self.display_images()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

        ttk.Button(self.param_frame, text="Применить разрешение", command=apply_res).pack(fill=tk.X)

    def show_pixelation_params(self):
        self.clear_param_frame()

        # размер пикселей
        ttk.Label(self.param_frame, text="15px",
                  font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        def apply_pixel():
            try:
                # центральная область
                h, w = self.original_image.shape[:2]
                x1, y1 = w // 4, h // 4
                x2, y2 = 3 * w // 4, 3 * h // 4

                self.current_image = apply_pixelation(
                    self.original_image.copy(),
                    (x1, y1, x2, y2)
                )
                self.display_images()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

        ttk.Button(self.param_frame, text="Применить пикселизацию", command=apply_pixel).pack(fill=tk.X)

    def show_simple_border_params(self):
        self.clear_param_frame()

        ttk.Label(self.param_frame, text="Ширина рамки:").pack(anchor=tk.W)
        width_var = tk.IntVar(value=20)
        width_scale = ttk.Scale(self.param_frame, from_=1, to=100, variable=width_var, orient=tk.HORIZONTAL)
        width_scale.pack(fill=tk.X, pady=(0, 10))

        value_label = ttk.Label(self.param_frame, text=f"Текущая ширина: {width_var.get()}px")
        value_label.pack(anchor=tk.W)

        def update_label(*args):
            value_label.config(text=f"Текущая ширина: {width_var.get()}px")

        width_var.trace('w', update_label)

        def apply_border():
            try:
                self.current_image = apply_simple_border(
                    self.original_image.copy(),
                    width_var.get()
                )
                self.display_images()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка применения рамки:\n{str(e)}")

        ttk.Button(self.param_frame, text="Применить черную рамку", command=apply_border).pack(fill=tk.X)

    def show_shaped_border_params(self):
        self.clear_param_frame()

        ttk.Label(self.param_frame, text="Тип фигурной рамки:").pack(anchor=tk.W)
        frame_type_var = tk.StringVar(value="wave")

        frame_types = [
            ("Волнистая", "wave"),
            ("Триугольная", "zigzag")
        ]

        for text, value in frame_types:
            ttk.Radiobutton(self.param_frame, text=text, variable=frame_type_var,
                            value=value).pack(anchor=tk.W)

        def apply_shaped_border_filter():
            try:
                self.current_image = apply_shaped_border(
                    self.original_image.copy(),
                    frame_type_var.get()
                )
                self.display_images()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка применения фигурной рамки:\n{str(e)}")

        ttk.Button(self.param_frame, text="Применить фигурную рамку",
                   command=apply_shaped_border_filter).pack(fill=tk.X, pady=(10, 0))

    def clear_param_frame(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

    def display_images(self):
        # Отображаем исходное изображение
        if self.original_image is not None:
            orig_display = self.resize_image_for_display(self.original_image, 400, 400)
            self.original_photo = ImageTk.PhotoImage(orig_display)
            self.original_canvas.delete("all")
            self.original_canvas.create_image(200, 200, image=self.original_photo, anchor=tk.CENTER)

        # Отображаем обработанное изображение
        if self.current_image is not None:
            proc_display = self.resize_image_for_display(self.current_image, 400, 400)
            self.processed_photo = ImageTk.PhotoImage(proc_display)
            self.processed_canvas.delete("all")
            self.processed_canvas.create_image(200, 200, image=self.processed_photo, anchor=tk.CENTER)

    def resize_image_for_display(self, image, max_width, max_height):
        h, w = image.shape[:2]

        #Вычисляем новые размеры с сохранением пропорций
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        #Изменяем размер изображения
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        #Конвертируем в PIL Image
        pil_image = Image.fromarray(resized.astype('uint8'))
        return pil_image


def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()