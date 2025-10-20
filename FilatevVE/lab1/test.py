import cv2
import numpy as np
import os
import glob


class OpenCVWindowManager:
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self.current_image = None
        self.original_image = None
        self.background_color = (0, 0, 0)
        self.last_window_size = None
        self.image_files = []
        self.current_image_index = 0
        self.image_names = []
        self.show_controls = True
        self.control_panel_height = 150

        # Кнопки управления
        self.buttons = []
        self.create_buttons()

        # Устанавливаем callback для обработки событий мыши
        cv2.setMouseCallback(window_name, self.on_mouse_event)

    def create_buttons(self):
        """Создать кнопки управления"""
        self.buttons = [
            {"label": "← Назад", "key": "b", "x": 10, "y": 10, "width": 80, "height": 30, "color": (70, 70, 200),
             "hover_color": (90, 90, 220)},
            {"label": "Вперед →", "key": "n", "x": 100, "y": 10, "width": 80, "height": 30, "color": (70, 200, 70),
             "hover_color": (90, 220, 90)},
            {"label": "Обновить", "key": "r", "x": 190, "y": 10, "width": 80, "height": 30, "color": (200, 200, 70),
             "hover_color": (220, 220, 90)},
            {"label": "Инфо", "key": "i", "x": 280, "y": 10, "width": 60, "height": 30, "color": (200, 120, 70),
             "hover_color": (220, 140, 90)},
            {"label": "Справка", "key": "h", "x": 350, "y": 10, "width": 70, "height": 30, "color": (120, 70, 200),
             "hover_color": (140, 90, 220)},
            {"label": "Панель", "key": "c", "x": 430, "y": 10, "width": 70, "height": 30, "color": (70, 200, 200),
             "hover_color": (90, 220, 220)},
            {"label": "Выход", "key": "q", "x": 510, "y": 10, "width": 60, "height": 30, "color": (200, 70, 70),
             "hover_color": (220, 90, 90)}
        ]

        # Информационные поля
        self.info_fields = [
            {"label": "Изображение:", "x": 10, "y": 50, "width": 300, "height": 20},
            {"label": "Размер окна:", "x": 10, "y": 75, "width": 200, "height": 20},
            {"label": "Размер изображения:", "x": 10, "y": 100, "width": 250, "height": 20},
            {"label": "Горячие клавиши:", "x": 10, "y": 125, "width": 400, "height": 20}
        ]

    def set_size(self, width, height):
        """Установить размер окна"""
        cv2.resizeWindow(self.window_name, width, height)

    def get_size(self):
        """Получить текущий размер окна"""
        props = cv2.getWindowImageRect(self.window_name)
        if props is not None:
            x, y, width, height = props
            return width, height
        return None

    def get_position(self):
        """Получить позицию окна"""
        props = cv2.getWindowImageRect(self.window_name)
        if props is not None:
            x, y, width, height = props
            return x, y
        return None

    def display_image(self, image):
        """Отобразить изображение в окне"""
        self.current_image = image.copy()

        # Добавляем панель управления если включено
        if self.show_controls:
            image_with_controls = self.add_control_panel(image)
            cv2.imshow(self.window_name, image_with_controls)
        else:
            cv2.imshow(self.window_name, image)

    def add_control_panel(self, image):
        """Добавить панель управления к изображению"""
        if image is None:
            return image

        img_height, img_width = image.shape[:2]

        # Создаем изображение с дополнительным местом для панели
        panel_height = self.control_panel_height
        combined_height = img_height + panel_height
        combined_image = np.full((combined_height, img_width, 3), (45, 45, 48), dtype=np.uint8)

        # Копируем оригинальное изображение
        combined_image[0:img_height, 0:img_width] = image

        # Добавляем панель управления
        self.draw_control_panel(combined_image[img_height:img_height + panel_height, 0:img_width])

        return combined_image

    def draw_control_panel(self, panel_area):
        """Нарисовать панель управления с кнопками"""
        panel_height, panel_width = panel_area.shape[:2]

        # Фон панели
        panel_area[:, :] = (45, 45, 48)

        # Разделительная линия
        cv2.line(panel_area, (0, 0), (panel_width, 0), (80, 80, 90), 2)

        # Рисуем кнопки
        for button in self.buttons:
            self.draw_button(panel_area, button)

        # Рисуем информационные поля
        self.draw_info_fields(panel_area)

        # Статус панели
        status_text = "Панель управления: ВКЛ" if self.show_controls else "Панель управления: ВЫКЛ"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(panel_area, status_text, (panel_width - text_size[0] - 10, panel_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)

    def draw_button(self, panel, button):
        """Нарисовать одну кнопку"""
        x, y, w, h = button["x"], button["y"], button["width"], button["height"]
        color = button["color"]

        # Рисуем прямоугольник кнопки
        cv2.rectangle(panel, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(panel, (x, y), (x + w, y + h), (200, 200, 200), 1)

        # Текст кнопки
        text_size = cv2.getTextSize(button["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(panel, button["label"], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Горячая клавиша
        key_text = f"({button['key'].upper()})"
        key_size = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        key_x = x + (w - key_size[0]) // 2
        key_y = y + h - 5
        cv2.putText(panel, key_text, (key_x, key_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    def draw_info_fields(self, panel):
        """Нарисовать информационные поля"""
        for field in self.info_fields:
            x, y = field["x"], field["y"]

            # Метка поля
            cv2.putText(panel, field["label"], (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Значения
            if field["label"] == "Изображение:":
                value = f"{self.get_current_image_name()} [{self.current_image_index + 1}/{len(self.image_files)}]"
                cv2.putText(panel, value, (x + 120, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)

            elif field["label"] == "Размер окна:":
                size = self.get_size()
                if size:
                    value = f"{size[0]} x {size[1]}"
                    cv2.putText(panel, value, (x + 120, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

            elif field["label"] == "Размер изображения:":
                if self.original_image is not None:
                    h, w = self.original_image.shape[:2]
                    value = f"{w} x {h}"
                    cv2.putText(panel, value, (x + 180, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)

            elif field["label"] == "Горячие клавиши:":
                value = "N, B, R, I, H, C, Q, F, +, -"
                cv2.putText(panel, value, (x + 140, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 200), 1)

    def wait_key(self, delay=0):
        """Ожидать нажатия клавиши"""
        return cv2.waitKey(delay) & 0xFF

    def on_mouse_event(self, event, x, y, flags, param):
        """Обработчик событий мыши для кнопок и изменения размера"""
        if not self.show_controls:
            return

        # Получаем размеры основного изображения
        window_size = self.get_size()
        if not window_size:
            return

        window_width, window_height = window_size
        img_height = window_height - self.control_panel_height

        # Корректируем координаты Y для панели управления
        if y > img_height:
            panel_y = y - img_height

            # Проверяем клики по кнопкам
            if event == cv2.EVENT_LBUTTONDOWN:
                for button in self.buttons:
                    if (button["x"] <= x <= button["x"] + button["width"] and
                            button["y"] <= panel_y <= button["y"] + button["height"]):
                        self.handle_button_click(button["key"])
                        break

            # Подсветка кнопок при наведении
            elif event == cv2.EVENT_MOUSEMOVE:
                for button in self.buttons:
                    if (button["x"] <= x <= button["x"] + button["width"] and
                            button["y"] <= panel_y <= button["y"] + button["height"]):
                        button["current_color"] = button["hover_color"]
                    else:
                        button["current_color"] = button["color"]
                # Обновляем отображение
                if self.original_image is not None:
                    self.add_image_with_padding(self.original_image)

        # Отслеживание изменения размера окна
        if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self.update_displayed_image_immediately()

    def handle_button_click(self, button_key):
        """Обработка нажатия кнопки"""
        print(f"Нажата кнопка: {button_key}")

        if button_key == 'n':
            self.show_next_image()
        elif button_key == 'b':
            self.show_previous_image()
        elif button_key == 'r':
            self.reload_current_image()
        elif button_key == 'i':
            self.show_image_info()
        elif button_key == 'h':
            self.print_help()
        elif button_key == 'c':
            self.toggle_control_panel()
        elif button_key == 'q':
            print("Выход по кнопке")
            cv2.destroyAllWindows()
            exit()

    def update_displayed_image_immediately(self):
        """Немедленное обновление изображения при изменении размера"""
        if self.original_image is not None:
            self._quick_add_image_with_padding(self.original_image, self.background_color)

    def _quick_add_image_with_padding(self, image, background_color=(0, 0, 0)):
        """Быстрое добавление изображения с сохранением пропорций"""
        window_size = self.get_size()
        if window_size is None or window_size == self.last_window_size:
            return

        self.last_window_size = window_size
        window_width, window_height = window_size
        img_height, img_width = image.shape[:2]

        window_ratio = window_width / window_height
        img_ratio = img_width / img_height

        # Вычислить новые размеры с сохранением пропорций
        if img_ratio > window_ratio:
            new_width = window_width
            new_height = int(window_width / img_ratio)
        else:
            new_height = window_height
            new_width = int(window_height * img_ratio)

        # Быстрое масштабирование
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((window_height, window_width, 3), background_color, dtype=np.uint8)

        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2

        padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

        self.display_image(padded_img)

    def add_image_with_padding(self, image, background_color=(0, 0, 0)):
        """Добавить изображение с сохранением пропорций и черными полосами"""
        if image is None:
            print("Ошибка: изображение не загружено")
            return

        # Сохраняем оригинальное изображение для будущих обновлений
        self.original_image = image.copy()
        self.background_color = background_color

        window_size = self.get_size()
        if window_size is None:
            print("Ошибка: не удалось получить размеры окна")
            return

        self.last_window_size = window_size
        window_width, window_height = window_size
        img_height, img_width = image.shape[:2]

        window_ratio = window_width / window_height
        img_ratio = img_width / img_height

        # Вычислить новые размеры с сохранением пропорций
        if img_ratio > window_ratio:
            new_width = window_width
            new_height = int(window_width / img_ratio)
        else:
            new_height = window_height
            new_width = int(window_height * img_ratio)

        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        padded_img = np.full((window_height, window_width, 3), background_color, dtype=np.uint8)

        y_offset = (window_height - new_height) // 2
        x_offset = (window_width - new_width) // 2

        padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

        self.display_image(padded_img)

        return padded_img

    def load_images_from_folder(self, folder_path="images"):
        """Загрузить все изображения из указанной папки"""
        if not os.path.exists(folder_path):
            print(f"Ошибка: папка '{folder_path}' не существует!")
            return False

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        self.image_files = []
        self.image_names = []

        for extension in extensions:
            self.image_files.extend(glob.glob(os.path.join(folder_path, extension)))
            self.image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))

        if not self.image_files:
            print(f"В папке '{folder_path}' не найдено изображений!")
            return False

        self.image_files.sort()
        for file_path in self.image_files:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            self.image_names.append(name_without_ext)

        print(f"Загружено {len(self.image_files)} изображений из папки '{folder_path}'")
        return True

    def get_current_image_name(self):
        """Получить название текущего изображения"""
        if self.current_image_index < len(self.image_names):
            return self.image_names[self.current_image_index]
        return ""

    def load_current_image(self):
        """Загрузить текущее изображение"""
        if not self.image_files:
            print("Нет загруженных изображений!")
            return None

        if self.current_image_index >= len(self.image_files):
            self.current_image_index = 0

        try:
            image = cv2.imread(self.image_files[self.current_image_index])
            if image is None:
                print(f"Ошибка загрузки изображения: {self.image_files[self.current_image_index]}")
                return None
            return image
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")
            return None

    def show_next_image(self):
        """Показать следующее изображение"""
        if not self.image_files:
            print("Нет загруженных изображений!")
            return False

        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        image = self.load_current_image()

        if image is not None:
            print(f"Текущее изображение: {self.get_current_image_name()}")
            self.add_image_with_padding(image)
            return True

        return False

    def show_previous_image(self):
        """Показать предыдущее изображение"""
        if not self.image_files:
            print("Нет загруженных изображений!")
            return False

        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        image = self.load_current_image()

        if image is not None:
            print(f"Текущее изображение: {self.get_current_image_name()}")
            self.add_image_with_padding(image)
            return True

        return False

    def reload_current_image(self):
        """Перезагрузить текущее изображение"""
        print("Перезагрузка текущего изображения")
        image = self.load_current_image()
        if image is not None:
            self.add_image_with_padding(image)

    def show_image_info(self):
        """Показать информацию об изображении"""
        if self.original_image is not None:
            img_height, img_width = self.original_image.shape[:2]
            print(f"\nИнформация об изображении:")
            print(f"  Название: {self.get_current_image_name()}")
            print(f"  Размер: {img_width}x{img_height}")
            print(f"  Файл: {os.path.basename(self.image_files[self.current_image_index])}")
            print(f"  Папка: {os.path.dirname(self.image_files[self.current_image_index])}")
        else:
            print("Изображение не загружено")

    def toggle_control_panel(self):
        """Переключить отображение панели управления"""
        self.show_controls = not self.show_controls
        status = "включена" if self.show_controls else "выключена"
        print(f"Панель управления {status}")

        if self.original_image is not None:
            self.add_image_with_padding(self.original_image)

    def print_help(self):
        """Показать справку по командам"""
        print("\n" + "=" * 50)
        print("СПРАВКА ПО УПРАВЛЕНИЮ:")
        print("=" * 50)
        print("Кнопки мыши или горячие клавиши:")
        print("  ← Назад (B)    - Предыдущее изображение")
        print("  Вперед → (N)   - Следующее изображение")
        print("  Обновить (R)   - Перезагрузить изображение")
        print("  Инфо (I)       - Информация об изображении")
        print("  Справка (H)    - Эта справка")
        print("  Панель (C)     - Вкл/Выкл панель управления")
        print("  Выход (Q)      - Закрыть программу")
        print("  F              - Полноэкранный режим")
        print("  +/-            - Увеличить/Уменьшить")
        print("=" * 50)


# Основная программа
if __name__ == "__main__":
    window = OpenCVWindowManager("OpenCV Image Viewer - Кнопки управления")
    window.set_size(1000, 700)

    # Загрузка изображений
    print("Загрузка изображений из папки 'images'...")

    if not window.load_images_from_folder("images"):
        print("Создание тестовых изображений...")
        os.makedirs("images", exist_ok=True)

        # Создаем тестовые изображения
        test_images = [
            ("wide_image_16_9.jpg", (400, 711, 3), 0),
            ("tall_image_9_16.jpg", (711, 400, 3), 1),
            ("square_image_1_1.jpg", (500, 500, 3), 2)
        ]

        for filename, size, channel in test_images:
            img = np.random.randint(0, 255, size, dtype=np.uint8)
            img[:, :, channel] = 255
            cv2.imwrite(f"images/{filename}", img)

        window.load_images_from_folder("images")

    # Показать первое изображение
    if window.image_files:
        image = window.load_current_image()
        if image is not None:
            print(f"Текущее изображение: {window.get_current_image_name()}")
            window.add_image_with_padding(image)
    else:
        print("Нет изображений для отображения!")
        exit()

    window.print_help()
    print("\nИспользуйте кнопки мыши или горячие клавиши для управления!")

    last_size = window.get_size()
    fullscreen = False

    while True:
        key = window.wait_key(10)

        # Проверяем изменение размера окна
        current_size = window.get_size()
        if current_size and current_size != last_size:
            window.update_displayed_image_immediately()
            last_size = current_size

        # Обработка горячих клавиш
        if key in [ord('n'), 83]:
            window.show_next_image()
        elif key in [ord('b'), 81]:
            window.show_previous_image()
        elif key == ord('r'):
            window.reload_current_image()
        elif key == ord('i'):
            window.show_image_info()
        elif key == ord('h'):
            window.print_help()
        elif key == ord('c'):
            window.toggle_control_panel()
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Полноэкранный режим")
            else:
                cv2.setWindowProperty(window.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Оконный режим")
        elif key in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()