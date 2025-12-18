#shaped_border
import numpy as np

def apply_shaped_border(image, frame_type='wave'):
    h, w = image.shape[:2]
    thickness = 15
    mask = np.zeros((h, w), dtype=np.uint8)

    x = np.arange(w)

    if frame_type == 'wave':
        # Верхняя и нижняя волнистые границы
        y_top = (np.sin(x / 20) * 10 + thickness).astype(np.int32)
        y_bottom = (np.sin(x / 20) * 10 + thickness).astype(np.int32)

        for i in range(w):
            mask[:y_top[i], i] = 1
            mask[h - y_bottom[i]:, i] = 1

    elif frame_type == 'zigzag':
        # Верхняя и нижняя зигзагообразные границы
        y_top = (np.abs((x % (thickness * 2)) - thickness) * 2 + thickness // 2).astype(np.int32)
        y_bottom = (np.abs((x % (thickness * 2)) - thickness) * 2 + thickness // 2).astype(np.int32)

        for i in range(w):
            mask[:y_top[i], i] = 1
            mask[h - y_bottom[i]:, i] = 1

    # Боковые границы
    mask[:, :thickness] = 1
    mask[:, -thickness:] = 1

    # Применяем цвет рамки
    frame = image.copy()
    colors = {
        'wave': (0, 255, 255),  # Желтый
        'zigzag': (0, 0, 255)  # Красный
    }

    color = colors.get(frame_type, (0, 255, 0))
    frame[mask > 0] = color

    return frame