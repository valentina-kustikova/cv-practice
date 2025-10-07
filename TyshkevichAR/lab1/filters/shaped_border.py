import numpy as np


def apply_shaped_border(image, frame_type='wave'):

    def add_figure_frame(img, color=(0, 255, 0), thickness=30, frame_type='wave'):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if frame_type == 'wave':
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            x = np.arange(w)
            if frame_type == 'wave':
                y = (np.sin(x / 20) * 10 + thickness).astype(np.int32)
                for i in range(w):
                    mask[:y[i], i] = 1
                    mask[-y[i]:, i] = 1
            mask[:, :thickness] = 1
            mask[:, -thickness:] = 1
            frame = img.copy()
            frame[mask > 0] = color
            return frame

        elif frame_type == 'zigzag':
            x = np.arange(w)
            y = (np.abs((x % (thickness)) - thickness // 2) * 3 + thickness // 2).astype(np.int32)
            for i in range(w):
                mask[:y[i], i] = 1
                mask[-y[i]:, i] = 1

        # Боковые границы
        mask[:, :thickness] = 1
        mask[:, -thickness:] = 1

        frame = img.copy()
        frame[mask > 0] = color
        return frame
    colors = {
        'wave': (0, 255, 255),
        'zigzag': (0, 0, 255)
    }

    color = colors.get(frame_type, (0, 255, 0))
    return add_figure_frame(image, color=color, thickness=15, frame_type=frame_type)