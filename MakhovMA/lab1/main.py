import argparse
import os
import sys

import cv2

from filters import (
    PixelateSelector,
    add_frame,
    add_lens_flare,
    add_paper_texture,
    apply_sepia,
    apply_vignette,
    create_sample_textures,
    resize_image,
)


AVAILABLE_FILTERS = ["resize", "sepia", "vignette", "pixelate", "frame", "flare", "texture"]


def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Применение фильтров к изображению")
    parser.add_argument("--gui", action="store_true", help="Запустить графический интерфейс")
    parser.add_argument("image", nargs="?", help="Путь к изображению для обработки")
    parser.add_argument(
        "filter",
        nargs="?",
        help="Тип фильтра (resize, sepia, vignette, pixelate, frame, flare, texture)",
    )
    parser.add_argument("params", nargs="*", help="Дополнительные параметры фильтра (опционально)")
    return parser.parse_args()


def ensure_default_texture(path_hint):
    """Создает папку и образцы текстур, если это необходимо."""
    if not path_hint:
        return

    directory = os.path.dirname(path_hint)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(path_hint):
        print("Создание недостающих текстур...")
        create_sample_textures()


def run_cli_mode(image_path, filter_name, params):
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка: не удалось загрузить изображение.")
        return

    ftype = filter_name.lower()
    if ftype not in AVAILABLE_FILTERS:
        print("Ошибка: неизвестный тип фильтра.")
        print("Доступные фильтры: resize, sepia, vignette, pixelate, frame, flare, texture")
        return

    params = params or []
    result = None

    if ftype == "resize":
        scale = float(params[0]) if params else 0.5
        result = resize_image(img, scale)
        print(f"Изменение размера: масштаб {scale}")

    elif ftype == "sepia":
        result = apply_sepia(img)
        print("Применен эффект сепии")

    elif ftype == "vignette":
        strength = float(params[0]) if params else 0.5
        result = apply_vignette(img, strength)
        print(f"Применена виньетка: сила {strength}")

    elif ftype == "pixelate":
        pixel_size = int(params[0]) if params else 10
        print("Выберите область для пикселизации:")
        print("- Зажмите левую кнопку мыши и выделите область")
        print("- Отпустите кнопку для применения эффекта")
        print("- Нажмите Enter для подтверждения или ESC для отмены")
        selector = PixelateSelector(img, pixel_size)
        result = selector.select_region()
        print("Пикселизация применена")

    elif ftype == "frame":
        thickness = int(params[0]) if params else 20
        color = (int(params[1]), int(params[2]), int(params[3])) if len(params) > 3 else (0, 0, 255)

        texture_type = "simple"
        texture_path = None
        frame_style = "wave"

        if len(params) > 4:
            texture_type = params[4]
            if texture_type == "fancy":
                if len(params) > 5:
                    texture_path = params[5]
                    if len(params) > 6:
                        frame_style = params[6]
                else:
                    texture_path = "textures/fancy_frame_texture.jpg"
                    ensure_default_texture(texture_path)

        result = add_frame(img, color, thickness, texture_type, texture_path, frame_style)
        print(f"Добавлена рамка: тип {texture_type}, толщина {thickness}")

    elif ftype == "flare":
        if not params:
            texture_path = "textures/flare_texture.jpg"
            ensure_default_texture(texture_path)
        else:
            texture_path = params[0]

        intensity = float(params[1]) if len(params) > 1 else 0.7
        result = add_lens_flare(img, texture_path, intensity)
        print(f"Добавлены блики: текстура {texture_path}, интенсивность {intensity}")

    elif ftype == "texture":
        if not params:
            texture_path = "textures/paper_texture.jpg"
            ensure_default_texture(texture_path)
        else:
            texture_path = params[0]

        intensity = float(params[1]) if len(params) > 1 else 0.3
        result = add_paper_texture(img, texture_path, intensity)
        print(f"Добавлена текстура бумаги: интенсивность {intensity}")

    if result is None:
        return

    input_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"result_{input_name}_{ftype}.jpg"
    cv2.imwrite(output_path, result)
    print(f"Результат сохранен как: {output_path}")

    try:
        cv2.imshow("Original", img)
        cv2.imshow("Result", result)
        print("Нажмите любую клавишу в окне для закрытия...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("Отображение окон не поддерживается в данной среде.")
        print(f"Результат сохранен в файл: {output_path}")


def launch_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, colorchooser
        from tkinter import ttk
    except ImportError as exc:
        print("Ошибка: для графического интерфейса требуется установленный Tkinter.")
        print(f"Подробности: {exc}")
        return

    try:
        from PIL import Image, ImageTk, ImageColor
    except ImportError as exc:
        print("Ошибка: для графического интерфейса требуется установить пакет Pillow (pip install pillow).")
        print(f"Подробности: {exc}")
        return

    class ImageFilterApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Фильтры изображений")
            self.root.minsize(960, 620)
            self.root.resizable(True, True)

            self.original_image = None
            self.result_image = None
            self.original_preview = None
            self.result_preview = None
            self.current_image_path = ""

            self.file_path_var = tk.StringVar(value="Изображение не выбрано")
            self.filter_var = tk.StringVar(value=AVAILABLE_FILTERS[0])
            self.param_values = {}
            self.preview_size_var = tk.IntVar(value=640)
            self.preview_size_label_var = tk.StringVar(value="640 px")

            self._build_ui()

        def _build_ui(self):
            main_frame = ttk.Frame(self.root, padding=10)
            main_frame.grid(row=0, column=0, sticky="nsew")
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)

            file_frame = ttk.Frame(main_frame)
            file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
            ttk.Button(file_frame, text="Открыть изображение", command=self.load_image).grid(
                row=0, column=0, padx=(0, 10)
            )
            ttk.Label(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, sticky="w")

            filter_frame = ttk.Frame(main_frame)
            filter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
            ttk.Label(filter_frame, text="Фильтр:").grid(row=0, column=0, sticky="w")
            filter_combo = ttk.Combobox(
                filter_frame, textvariable=self.filter_var, values=AVAILABLE_FILTERS, state="readonly", width=20
            )
            filter_combo.grid(row=0, column=1, padx=(10, 0), sticky="w")
            filter_combo.bind("<<ComboboxSelected>>", self.on_filter_change)

            self.params_container = ttk.Frame(main_frame)
            self.params_container.grid(row=2, column=0, columnspan=2, sticky="ew")
            self.on_filter_change()

            preview_controls = ttk.Frame(main_frame)
            preview_controls.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 4))
            preview_controls.columnconfigure(1, weight=1)
            ttk.Label(preview_controls, text="Размер предпросмотра:").grid(row=0, column=0, sticky="w")
            size_scale = ttk.Scale(
                preview_controls,
                from_=240,
                to=960,
                variable=self.preview_size_var,
                orient="horizontal",
                command=lambda _evt: self.on_preview_size_change(),
            )
            size_scale.grid(row=0, column=1, sticky="ew", padx=10)
            ttk.Label(preview_controls, textvariable=self.preview_size_label_var).grid(row=0, column=2, padx=(6, 0))

            preview_frame = ttk.Frame(main_frame)
            preview_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="nsew")
            main_frame.rowconfigure(4, weight=1)
            preview_frame.columnconfigure(0, weight=1)
            preview_frame.columnconfigure(1, weight=1)

            self.original_image_label = tk.Label(
                preview_frame,
                text="Оригинал\n(нет изображения)",
                relief="groove",
                anchor="center",
                justify="center",
                padx=10,
                pady=10,
            )
            self.original_image_label.grid(row=0, column=0, padx=(0, 10))

            self.result_image_label = tk.Label(
                preview_frame,
                text="Результат\n(пока не применен)",
                relief="groove",
                anchor="center",
                justify="center",
                padx=10,
                pady=10,
            )
            self.result_image_label.grid(row=0, column=1)

            actions_frame = ttk.Frame(main_frame)
            actions_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0), sticky="ew")

            self.apply_button = ttk.Button(actions_frame, text="Применить фильтр", command=self.apply_filter)
            self.apply_button.grid(row=0, column=0, padx=(0, 10))

            self.save_button = ttk.Button(actions_frame, text="Сохранить результат", command=self.save_result, state="disabled")
            self.save_button.grid(row=0, column=1)

        def on_filter_change(self, _event=None):
            for child in self.params_container.winfo_children():
                child.destroy()
            self.param_values = {}

            selected = self.filter_var.get()
            row_idx = 0

            if selected == "resize":
                row_idx = self._make_double_param("Масштаб", "scale", 0.5, 0.1, 3.0, 0.1, format_str="%.2f", row=row_idx)
            elif selected == "vignette":
                row_idx = self._make_double_param(
                    "Сила виньетки", "strength", 0.5, 0.0, 1.0, 0.05, format_str="%.2f", row=row_idx
                )
            elif selected == "pixelate":
                row_idx = self._make_int_param("Размер пикселя", "pixel_size", 10, 2, 150, 1, row=row_idx)
            elif selected == "frame":
                row_idx = self._make_int_param("Толщина рамки", "thickness", 20, 1, 200, 1, row=row_idx)
                color_var = tk.StringVar(value="#ff0000")
                self.param_values["color"] = color_var
                color_frame = ttk.Frame(self.params_container)
                color_frame.grid(row=row_idx, column=0, sticky="w", pady=2)
                ttk.Label(color_frame, text="Цвет рамки:").grid(row=0, column=0, sticky="w")
                color_preview = tk.Label(color_frame, width=3, background=color_var.get())
                color_preview.grid(row=0, column=1, padx=5)
                ttk.Button(
                    color_frame,
                    text="Выбрать...",
                    command=lambda: self._choose_color(color_var, color_preview),
                ).grid(row=0, column=2, padx=(5, 0))
                row_idx += 1

                texture_type_var = tk.StringVar(value="simple")
                self.param_values["texture_type"] = texture_type_var
                ttk.Label(self.params_container, text="Тип текстуры:").grid(row=row_idx, column=0, sticky="w", pady=(4, 0))
                texture_combo = ttk.Combobox(
                    self.params_container, textvariable=texture_type_var, values=["simple", "fancy"], state="readonly"
                )
                texture_combo.grid(row=row_idx, column=0, sticky="ew")
                row_idx += 1

                texture_path_var = tk.StringVar(value="")
                self.param_values["texture_path"] = texture_path_var
                texture_frame = ttk.Frame(self.params_container)
                texture_frame.grid(row=row_idx, column=0, sticky="ew", pady=2)
                ttk.Label(texture_frame, text="Путь к текстуре:").grid(row=0, column=0, sticky="w")
                texture_entry = ttk.Entry(texture_frame, textvariable=texture_path_var, width=40)
                texture_entry.grid(row=0, column=1, padx=5)
                ttk.Button(
                    texture_frame,
                    text="Обзор...",
                    command=lambda: self._browse_texture(texture_path_var),
                ).grid(row=0, column=2)
                row_idx += 1

                frame_style_var = tk.StringVar(value="wave")
                self.param_values["frame_style"] = frame_style_var
                ttk.Label(self.params_container, text="Стиль рамки:").grid(row=row_idx, column=0, sticky="w", pady=(4, 0))
                frame_style_combo = ttk.Combobox(
                    self.params_container, textvariable=frame_style_var, values=["wave", "zigzag"], state="readonly"
                )
                frame_style_combo.grid(row=row_idx, column=0, sticky="ew")
                row_idx += 1

            elif selected == "flare":
                texture_path_var = tk.StringVar(value="")
                self.param_values["texture_path"] = texture_path_var
                texture_frame = ttk.Frame(self.params_container)
                texture_frame.grid(row=row_idx, column=0, sticky="ew", pady=2)
                ttk.Label(texture_frame, text="Текстура бликов:").grid(row=0, column=0, sticky="w")
                ttk.Entry(texture_frame, textvariable=texture_path_var, width=40).grid(row=0, column=1, padx=5)
                ttk.Button(
                    texture_frame,
                    text="Обзор...",
                    command=lambda: self._browse_texture(texture_path_var),
                ).grid(row=0, column=2)
                row_idx += 1
                row_idx = self._make_double_param(
                    "Интенсивность", "intensity", 0.7, 0.0, 2.0, 0.1, format_str="%.2f", row=row_idx
                )

            elif selected == "texture":
                texture_path_var = tk.StringVar(value="")
                self.param_values["texture_path"] = texture_path_var
                texture_frame = ttk.Frame(self.params_container)
                texture_frame.grid(row=row_idx, column=0, sticky="ew", pady=2)
                ttk.Label(texture_frame, text="Текстура бумаги:").grid(row=0, column=0, sticky="w")
                ttk.Entry(texture_frame, textvariable=texture_path_var, width=40).grid(row=0, column=1, padx=5)
                ttk.Button(
                    texture_frame,
                    text="Обзор...",
                    command=lambda: self._browse_texture(texture_path_var),
                ).grid(row=0, column=2)
                row_idx += 1
                row_idx = self._make_double_param(
                    "Интенсивность", "intensity", 0.3, 0.0, 1.0, 0.05, format_str="%.2f", row=row_idx
                )

            else:
                ttk.Label(self.params_container, text="Дополнительные параметры не требуются.").grid(row=0, column=0, sticky="w")
        
        def refresh_previews(self):
            if self.original_image is not None:
                self._update_preview(self.original_image, target="original")
            if self.result_image is not None:
                self._update_preview(self.result_image, target="result")

        def on_preview_size_change(self):
            size = int(self.preview_size_var.get())
            self.preview_size_label_var.set(f"{size} px")
            self.refresh_previews()

        def _make_double_param(
            self, label, key, default, min_value, max_value, increment, format_str="%.2f", row=0
        ):
            var = tk.DoubleVar(value=default)
            frame = ttk.Frame(self.params_container)
            frame.grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(frame, text=f"{label} ({min_value}..{max_value}):").grid(row=0, column=0, sticky="w")
            spin = ttk.Spinbox(
                frame,
                from_=min_value,
                to=max_value,
                increment=increment,
                textvariable=var,
                width=8,
                format=format_str,
            )
            spin.grid(row=0, column=1, padx=(5, 0))
            self.param_values[key] = var
            return row + 1

        def _make_int_param(self, label, key, default, min_value, max_value, increment, row=0):
            var = tk.IntVar(value=default)
            frame = ttk.Frame(self.params_container)
            frame.grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(frame, text=f"{label} ({min_value}..{max_value}):").grid(row=0, column=0, sticky="w")
            spin = ttk.Spinbox(
                frame,
                from_=min_value,
                to=max_value,
                increment=increment,
                textvariable=var,
                width=6,
            )
            spin.grid(row=0, column=1, padx=(5, 0))
            self.param_values[key] = var
            return row + 1

        def _choose_color(self, color_var, preview_label):
            color = colorchooser.askcolor(color=color_var.get())
            if color[1]:
                color_var.set(color[1])
                preview_label.configure(background=color[1])

        def _browse_texture(self, var):
            file_path = filedialog.askopenfilename(
                title="Выберите файл текстуры",
                filetypes=[
                    ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("Все файлы", "*.*"),
                ],
            )
            if file_path:
                var.set(file_path)

        def load_image(self):
            file_path = filedialog.askopenfilename(
                title="Выберите изображение",
                filetypes=[
                    ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("Все файлы", "*.*"),
                ],
            )
            if not file_path:
                return

            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение.")
                return

            self.original_image = image
            self.result_image = None
            self.current_image_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            self._update_preview(image, target="original")
            self.result_image_label.configure(text="Результат\n(пока не применен)", image="")
            self.result_preview = None
            self.save_button.state(["disabled"])

        def apply_filter(self):
            if self.original_image is None:
                messagebox.showinfo("Информация", "Пожалуйста, откройте изображение перед применением фильтра.")
                return

            filter_name = self.filter_var.get()
            try:
                result = self._process_filter(filter_name)
            except Exception as exc:
                messagebox.showerror("Ошибка", f"Не удалось применить фильтр:\n{exc}")
                return

            if result is not None:
                self.result_image = result
                self._update_preview(result, target="result")
                self.save_button.state(["!disabled"])

        def _process_filter(self, filter_name):
            image = self.original_image.copy()

            if filter_name == "resize":
                scale = max(0.01, float(self.param_values["scale"].get()))
                return resize_image(image, scale)

            if filter_name == "sepia":
                return apply_sepia(image)

            if filter_name == "vignette":
                strength = float(self.param_values["strength"].get())
                strength = max(0.0, min(1.5, strength))
                return apply_vignette(image, strength)

            if filter_name == "pixelate":
                pixel_size = max(1, int(self.param_values["pixel_size"].get()))
                self.root.withdraw()
                try:
                    selector = PixelateSelector(image, pixel_size)
                    result = selector.select_region()
                finally:
                    self.root.deiconify()
                    self.root.update()
                return result

            if filter_name == "frame":
                thickness = max(1, int(self.param_values["thickness"].get()))
                color_hex = self.param_values["color"].get()
                r, g, b = ImageColor.getrgb(color_hex)
                color_bgr = (b, g, r)

                texture_type = self.param_values["texture_type"].get()
                texture_path = self.param_values["texture_path"].get().strip() or None

                if texture_type == "fancy" and not texture_path:
                    texture_path = "textures/fancy_frame_texture.jpg"
                    ensure_default_texture(texture_path)
                elif texture_path:
                    ensure_default_texture(texture_path)

                frame_style = self.param_values["frame_style"].get()
                return add_frame(image, color_bgr, thickness, texture_type, texture_path, frame_style)

            if filter_name == "flare":
                texture_path = self.param_values["texture_path"].get().strip()
                if not texture_path:
                    texture_path = "textures/flare_texture.jpg"
                ensure_default_texture(texture_path)
                intensity = float(self.param_values["intensity"].get())
                return add_lens_flare(image, texture_path, intensity)

            if filter_name == "texture":
                texture_path = self.param_values["texture_path"].get().strip()
                if not texture_path:
                    texture_path = "textures/paper_texture.jpg"
                ensure_default_texture(texture_path)
                intensity = float(self.param_values["intensity"].get())
                return add_paper_texture(image, texture_path, intensity)

            raise ValueError("Неизвестный фильтр.")

        def _update_preview(self, image, target):
            if image is None:
                return

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            max_size = max(1, int(self.preview_size_var.get()))
            pil_image.thumbnail((max_size, max_size))
            photo = ImageTk.PhotoImage(pil_image)

            if target == "original":
                self.original_image_label.configure(image=photo, text="")
                self.original_image_label.image = photo
                self.original_preview = photo
            else:
                self.result_image_label.configure(image=photo, text="")
                self.result_image_label.image = photo
                self.result_preview = photo

        def save_result(self):
            if self.result_image is None:
                messagebox.showinfo("Информация", "Нет результата для сохранения.")
                return

            default_name = "result.jpg"
            if self.current_image_path:
                base = os.path.splitext(os.path.basename(self.current_image_path))[0]
                default_name = f"result_{base}_{self.filter_var.get()}.jpg"

            file_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                initialfile=default_name,
                filetypes=[
                    ("JPEG", "*.jpg;*.jpeg"),
                    ("PNG", "*.png"),
                    ("BMP", "*.bmp"),
                    ("All files", "*.*"),
                ],
            )
            if not file_path:
                return

            try:
                cv2.imwrite(file_path, self.result_image)
            except Exception as exc:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{exc}")
                return

            messagebox.showinfo("Готово", f"Результат сохранен: {file_path}")

    root = tk.Tk()
    ImageFilterApp(root)
    root.mainloop()


def main():
    args = parse_cli_arguments()

    if args.gui or not args.image or not args.filter:
        launch_gui()
        return

    run_cli_mode(args.image, args.filter, args.params)


if __name__ == "__main__":
    main()
