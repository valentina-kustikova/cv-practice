from lib.change_images import *


def change_sepia(window):
    """
    Менеджер для функции применения фотоэффекта сепии к изображению.
    """
    current_value = 1
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_value
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 and current_value <= 0.9:
                current_value += 0.05
            elif flags < 0 and current_value >= 0.1:
                current_value -= 0.05

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == 27:
            break

        image = window.load_current_image()
        new_image = sepia_filter(image, current_value)
        window.add_image_with_padding(new_image)

    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())


def change_apply_vignette_elliptical(window):
    """
    Менеджер для функции применения фотоэффекта виньетки к изображению.
    """
    current_scale = 1.5
    current_intensity = 0.7

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_scale
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 and current_scale <= 4.5:
                current_scale += 0.1
            elif flags < 0 and current_scale >= 1.0:
                current_scale -= 0.1

    cv2.setMouseCallback(window.window, mouse_callback)


    while True:
        key = window.wait_key(60)

        if key == ord('w') and current_intensity <= 0.9:
            current_intensity += 0.1

        elif key == ord('s') and current_intensity >= 0.1:
            current_intensity -= 0.1

        elif key == 27:
            break

        image = window.load_current_image()
        new_image = apply_vignette_elliptical(image, current_scale, current_intensity)
        window.add_image_with_padding(new_image)

    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())


def change_add_lens_flare(window):
    flare_position = (0.5, 0.5)
    intensity = 1.0
    flare_size = 0.3

    def mouse_callback(event, x, y, flags, param):
        nonlocal intensity
        nonlocal flare_position
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 and intensity <= 0.9:
                intensity += 0.05
            elif flags < 0 and intensity >= 0.1:
                intensity -= 0.05
        elif event == cv2.EVENT_LBUTTONDOWN:
            window_size = window.get_size()
            flare_position = (x / window_size[0], y / window_size[1])

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == ord('w') and flare_size <= 0.9:
            flare_size += 0.1

        elif key == ord('s') and flare_size >= 0.2:
            flare_size -= 0.1

        elif key == 27:
            break

        image = window.load_current_image()
        new_image = add_lens_flare(image, flare_position, intensity, flare_size)
        window.add_image_with_padding(new_image)


    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())

def change_add_watercolor_texture(window):
    current_value = 1
    last_value = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_value
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 and current_value <= 0.9:
                current_value += 0.05
            elif flags < 0 and current_value >= 0.1:
                current_value -= 0.05

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == 27:
            break

        if last_value != current_value:
            image = window.load_current_image()
            new_image = add_watercolor_texture(image, current_value)
            window.add_image_with_padding(new_image)
            last_value = current_value

    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())


def changer_add_rectangular_border(window):
    color = [100, 150, 200]
    current_channel = 0
    border_width = 10

    def mouse_callback(event, x, y, flags, param):
        nonlocal color
        nonlocal current_channel
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                color[current_channel] = min(color[current_channel] + 5, 255)
            elif flags < 0:
                color[current_channel] = max(color[current_channel] - 5, 0)
        elif event == cv2.EVENT_LBUTTONDOWN:
            current_channel = (current_channel + 1) % 3

    cv2.setMouseCallback(window.window, mouse_callback)

    while True:
        key = window.wait_key(60)

        if key == ord('w'):
            border_width += 1

        elif key == ord('s') and border_width >= 1:
            border_width -= 1

        if key == 27:
            break

        image = window.load_current_image()
        new_image = add_rectangular_border(image, border_width, tuple(color))
        window.add_image_with_padding(new_image)

    cv2.setMouseCallback(window.window_name, lambda *args: None)
    window.add_image_with_padding(window.load_current_image())