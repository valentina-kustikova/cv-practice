import cv2
import numpy as np

def resize_image(image, width=None, height=None, scale_factor=None):
	"""
	Изменение разрешения изображения
	Args:
		image: Входное изображение
		width: Новая ширина
		height: Новая высота
		scale_factor: Коэффициент масштабирования
	Returns:
		Изображение с измененным разрешением
	"""
	h, w = image.shape[:2]
	if scale_factor is not None:
		new_width = max(int(w * scale_factor), 1)
		new_height = max(int(h * scale_factor), 1)
	elif width is not None and height is not None:
		new_width = width
		new_height = height
	elif width is not None:
		ratio = width / w
		new_width = width
		new_height = max(int(h * ratio), 1)
	elif height is not None:
		ratio = height / h
		new_width = max(int(w * ratio), 1)
		new_height = height
	else:
		return image.copy()
	# Создаем массив для нового изображения
	resized = np.zeros((new_height, new_width, 3), dtype=image.dtype)
	# Создаем координатные сетки для исходного изображения
	y_ratio = np.linspace(0, h - 1, new_height).astype(int)
	x_ratio = np.linspace(0, w - 1, new_width).astype(int)
	y_indices, x_indices = np.meshgrid(y_ratio, x_ratio, indexing='ij')
	resized = image[y_indices, x_indices]
	return resized

def sepia(img):
	"""
	Фотоэффект сепии через базовые матричные операции.
	"""
	kernel = np.array([[0.272, 0.534, 0.131],
					  [0.349, 0.686, 0.168],
					  [0.393, 0.769, 0.189]])
	sepia_img = img.copy().astype(np.float32)
	sepia_img = np.dot(sepia_img, kernel.T)
	sepia_img = np.clip(sepia_img, 0, 255)
	return sepia_img.astype(np.uint8)

def vignette(img, strength=0.5):
	"""
	Фотоэффект виньетки через базовые операции.
	"""
	rows, cols = img.shape[:2]
	X_result = np.linspace(-1, 1, cols)
	Y_result = np.linspace(-1, 1, rows)
	X, Y = np.meshgrid(X_result, Y_result)
	mask = 1 - strength * (X**2 + Y**2)
	mask = np.clip(mask, 0, 1)
	vignette_img = img.astype(np.float32)
	mask_3d = mask[..., np.newaxis]
	vignette_img *= mask_3d
	return np.clip(vignette_img, 0, 255).astype(np.uint8)

def pixelate_area(img, x1, y1, x2, y2, block_size=10):
	"""
	Пикселизация заданной прямоугольной области.
	"""
	out = img.copy()
	roi = out[y1:y2, x1:x2]
	h, w = roi.shape[:2]

	if h == 0 or w == 0:
		return out

	temp = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_NEAREST)
	pixelated_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
	out[y1:y2, x1:x2] = pixelated_roi
	return out

def add_rect_border(img, color=(0, 255, 0), thickness=10):
	"""
	Прямоугольная одноцветная рамка по краям изображения.
	"""
	out = img.copy()
	h, w = img.shape[:2]
	out[:thickness, :] = color
	out[h-thickness:, :] = color
	out[:, :thickness] = color
	out[:, w-thickness:] = color
	return out

def add_shape_border(img, color=(255, 0, 0), thickness=10, shape='circle'):
    """
    Фигурная одноцветная рамка по краям изображения.
    shape: 'circle', 'star', 'wave', ...
    """
    out = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    line_type = cv2.LINE_AA

    if shape == 'circle':
        cv2.circle(mask, (w//2, h//2), min(w, h)//2 - thickness, 255, -1, lineType=line_type)
    elif shape == 'star':
        points = [
            (w//2, thickness), (w//2 + w//6, h//2 - h//6),
            (w - thickness, h//2), (w//2 + w//6, h//2 + h//6),
            (w//2, h - thickness), (w//2 - w//6, h//2 + h//6),
            (thickness, h//2), (w//2 - w//6, h//2 - h//6)
        ]
        cv2.fillPoly(mask, [np.array(points)], 255, lineType=line_type)
    elif shape == 'wave':
        points = []
        amplitude = thickness * 1.5
        num_waves_w = 6
        num_waves_h = 5
        fade_length = min(w, h) / (max(num_waves_w, num_waves_h) * 1.5)
        freq_w = 2 * np.pi * num_waves_w / w
        freq_h = 2 * np.pi * num_waves_h / h

        def get_fade_factor(pos, total_len, fade_len):
            if pos < fade_len:
                return pos / fade_len
            elif pos > total_len - fade_len:
                return (total_len - pos) / fade_len
            return 1.0

        for x in range(w):
            fade = get_fade_factor(x, w, fade_length)
            y_offset = int(amplitude * fade * np.sin(freq_w * x))
            points.append((x, thickness + y_offset))
        for y in range(h):
            fade = get_fade_factor(y, h, fade_length)
            x_offset = int(amplitude * fade * np.sin(freq_h * y))
            points.append((w - thickness - x_offset, y))
        for x in range(w - 1, -1, -1):
            fade = get_fade_factor(x, w, fade_length)
            y_offset = int(amplitude * fade * np.sin(freq_w * x))
            points.append((x, h - thickness - y_offset))
        for y in range(h - 1, -1, -1):
            fade = get_fade_factor(y, h, fade_length)
            x_offset = int(amplitude * fade * np.sin(freq_h * y))
            points.append((thickness + x_offset, y))
        cv2.fillPoly(mask, [np.array(points)], 255, lineType=line_type)

    out[mask == 0] = color
    return out

def lens_flare(image, center_x=None, center_y=None, intensity=0.8):
	"""
	Наложение эффекта бликов объектива камеры
	Args:
		image: входное изображение
		center_x: x-координата центра блика
		center_y: y-координата центра блика
		intensity: интенсивность блика (0.0 - 1.0)
	Returns:
		изображение с эффектом бликов
	"""
	h, w = image.shape[:2]
	result = image.astype(np.float32)
	if center_x is None:
		center_x = w * 0.7
	if center_y is None:
		center_y = h * 0.3
	y_coords, x_coords = np.ogrid[:h, :w]
	radius_main = min(w, h) // 4
	distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
	mask_main = distance < radius_main
	flare_strength = np.zeros((h, w))
	flare_strength[mask_main] = (1.0 - distance[mask_main] / radius_main) * intensity
	flare_strength_3d = flare_strength[:, :, np.newaxis]
	result = np.minimum(255, result + flare_strength_3d * 200)
	num_artifacts = 3
	for k in range(num_artifacts):
		artifact_x = center_x - (center_x - w / 2) * (k + 1) / (num_artifacts + 1)
		artifact_y = center_y - (center_y - h / 2) * (k + 1) / (num_artifacts + 1)
		artifact_radius = radius_main // (k + 3)
		distance_artifact = np.sqrt((x_coords - artifact_x)**2 + (y_coords - artifact_y)**2)
		mask_artifact = distance_artifact < artifact_radius
		artifact_strength = np.zeros((h, w))
		artifact_strength[mask_artifact] = (1.0 - distance_artifact[mask_artifact] / artifact_radius) * intensity * 0.3
		channel = k % 3
		result[:, :, channel] = np.minimum(255, result[:, :, channel] + artifact_strength * 150)
	return result.astype(np.uint8)

def watercolor_texture(img, strength=0.3):
	"""
	Эффект текстуры акварельной бумаги.
	"""
	out = img.astype(np.float32)
	h, w = img.shape[:2]
	noise = np.random.normal(loc=128, scale=40, size=(h, w)).astype(np.float32)
	noise = cv2.GaussianBlur(noise, (9, 9), 0)
	noise_3d = noise[..., np.newaxis]
	out = out * (1 - strength) + noise_3d * strength
	return np.clip(out, 0, 255).astype(np.uint8)
