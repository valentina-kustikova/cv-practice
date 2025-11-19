import cv2
import numpy as np

# В этом файле хранятся только функции-фильтры

def resize_image(image, width, height):
    """1. Изменяет разрешение изображения."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def apply_sepia(image):
    """2. Применяет фотоэффект сепия к изображению."""
    img_sepia = image.copy()
    img_sepia = np.array(img_sepia, dtype=np.float64)
    sepia_matrix = np.matrix([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
    img_sepia = cv2.transform(img_sepia, sepia_matrix)
    img_sepia[np.where(img_sepia > 255)] = 255
    return np.array(img_sepia, dtype=np.uint8)

def apply_vignette(image, strength=0.8):
    """3. Применяет эффект виньетки (затемнение углов)."""
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, int(cols * strength))
    kernel_y = cv2.getGaussianKernel(rows, int(rows * strength))
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    output = image.copy()
    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask
    return output

def pixelate_area(image, x, y, w, h, pixel_size=10):
    """4. Пикселизирует выбранную прямоугольную область."""
    output = image.copy()
    roi = output[y:y+h, x:x+w]
    roi_h, roi_w = roi.shape[:2]
    temp = cv2.resize(roi, (roi_w // pixel_size, roi_h // pixel_size), interpolation=cv2.INTER_NEAREST)
    pixelated_roi = cv2.resize(temp, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    output[y:y+h, x:x+w] = pixelated_roi
    return output

def add_simple_frame(image, thickness, color):
    """5. Добавляет простую одноцветную прямоугольную рамку."""
    output = image.copy()
    h, w = output.shape[:2]
    output[0:thickness, :] = color
    output[h-thickness:h, :] = color
    output[:, 0:thickness] = color
    output[:, w-thickness:w] = color
    return output

def add_image_frame(image, frame_path):
    """6. Накладывает рамку из PNG файла с прозрачностью."""
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(f"Не удалось загрузить файл рамки: {frame_path}")

    # Подгоняем размер рамки под размер изображения
    h, w = image.shape[:2]
    frame = cv2.resize(frame, (w, h))

    # Разделяем RGB и альфа-канал
    frame_rgb = frame[:, :, :3]
    alpha = frame[:, :, 3] / 255.0
    
    # Создаем 3-канальную альфа-маску для умножения
    alpha_mask = cv2.merge([alpha, alpha, alpha])

    # Альфа-блендинг: Result = Frame * Alpha + Image * (1 - Alpha)
    blended = frame_rgb * alpha_mask + image * (1 - alpha_mask)
    
    return np.clip(blended, 0, 255).astype(np.uint8)

def add_image_flare(image, flare_path, center_x, center_y, scale=1.0):
    """7. Добавляет блик из файла с возможностью указания центра и масштаба."""
    flare = cv2.imread(flare_path)
    if flare is None:
        raise FileNotFoundError(f"Не удалось загрузить файл блика: {flare_path}")

    # Масштабируем блик
    flare_h, flare_w = flare.shape[:2]
    flare = cv2.resize(flare, (int(flare_w * scale), int(flare_h * scale)))
    
    # Получаем новые размеры блика после масштабирования
    flare_h, flare_w = flare.shape[:2]
    
    # Вычисляем, где на исходном изображении будет располагаться блик
    # top-left corner of the flare
    x0 = center_x - flare_w // 2
    y0 = center_y - flare_h // 2
    
    # Убедимся, что блик не выходит за границы изображения
    # Находим область пересечения блика и основного изображения
    img_h, img_w = image.shape[:2]
    overlap_x0 = max(0, x0)
    overlap_y0 = max(0, y0)
    overlap_x1 = min(img_w, x0 + flare_w)
    overlap_y1 = min(img_h, y0 + flare_h)

    # Если нет пересечения, просто возвращаем исходное изображение
    if overlap_x0 >= overlap_x1 or overlap_y0 >= overlap_y1:
        return image

    # Вырезаем область (Region of Interest - ROI) из основного изображения
    image_roi = image[overlap_y0:overlap_y1, overlap_x0:overlap_x1]

    # Вырезаем соответствующую часть из самого блика
    flare_x0_in_flare = overlap_x0 - x0
    flare_y0_in_flare = overlap_y0 - y0
    flare_roi = flare[flare_y0_in_flare:flare_y0_in_flare + image_roi.shape[0], 
                      flare_x0_in_flare:flare_x0_in_flare + image_roi.shape[1]]

    # Применяем режим наложения "Screen" только к этой области
    image_roi_float = image_roi.astype(np.float32) / 255.0
    flare_roi_float = flare_roi.astype(np.float32) / 255.0
    blended_roi_float = 1 - (1 - image_roi_float) * (1 - flare_roi_float)
    blended_roi = np.clip(blended_roi_float * 255, 0, 255).astype(np.uint8)

    # Создаем копию исходного изображения и вставляем обработанную область
    output = image.copy()
    output[overlap_y0:overlap_y1, overlap_x0:overlap_x1] = blended_roi
    
    return output

def apply_paper_texture(image, paper_path, strength=0.4):
    """8. Накладывает текстуру бумаги."""
    paper = cv2.imread(paper_path)
    if paper is None:
        raise FileNotFoundError(f"Не удалось загрузить файл текстуры: {paper_path}")

    # Подгоняем размер текстуры под размер изображения
    h, w = image.shape[:2]
    paper = cv2.resize(paper, (w, h))

    # Простое смешивание с помощью addWeighted
    # Result = image * (1 - strength) + paper * strength
    return cv2.addWeighted(image, 1 - strength, paper, strength, 0)