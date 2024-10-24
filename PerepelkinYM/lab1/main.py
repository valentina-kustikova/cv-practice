import numpy as np
import cv2 as cv
import argparse


def cli_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('--image', '-i',
						help='Image path',
						type=str,
						dest='image_path',
						required=False,
						default="image.png")

	parser.add_argument("--filter", "-f",
						help="Filter type (grey, resize, sepia, vignette, pixelate)",
						type=str,
						dest="filter_type",
						required=False,
						default="pixelate",
						choices=["grey", "resize", "sepia", "vignette", "pixelate"])

	# Аргументы для фильтра resize
	parser.add_argument("--width", "-w",
						help="Width for resizing",
						type=int,
						dest="width",
						required=False,
						default=1920)

	parser.add_argument("--height", "-ht",
						help="Height for resizing",
						type=int,
						dest="height",
						required=False,
						default=1080)

	# Аргумент для фильтра vignette
	parser.add_argument("--radius", "-r",
						help="Radius for vignette",
						type=int,
						dest="radius",
						required=False,
						default=80)

	# Аргумент для фильтра pixelate
	parser.add_argument("--pixel_size", "-ps",
						help="Pixel size for pixelate",
						type=int,
						dest="pixel_size",
						required=False,
						default=10)

	return parser.parse_args()


def image_grey(img):
	"""
	Функция перевода изображения в оттенки серого

	:param img: Исходное изображение
	:return: Изменённое изображение в оттенках серого
	"""

	weights = np.array([0.1140, 0.5870, 0.2989])                # Параметры инвертированы, т.к. opencv хранит изображения в BGR
	res_img = np.dot(img[:, :, :3], weights).astype(np.uint8)   # img[..., :3] = img[:,:, :3] - игнорируем альфа канал

	return res_img


def image_change_resolution(img, new_width, new_height):
	"""
	Функция изменения разрешения изображения

	:param img: Исходное изображение
	:param new_width: Ширина нового изображения
	:param new_height: Высота нового изображения
	:return: Изменённое изображение с новым разрешением
	"""

	if new_width <= 0 or new_height <= 0:
		raise ValueError("Invalid width or height.")
	old_height, old_width = img.shape[:2]

	# Коэффициенты масштабирования
	x_ratio, y_ratio = old_width / new_width, old_height / new_height

	# Создаем сетку координат для нового изображения
	# Пиксель из оригинального изображения для каждого пикселя нового изображения
	x_indices = (np.arange(new_width) * x_ratio).astype(np.int32)       # np.int32, так как координаты
	y_indices = (np.arange(new_height) * y_ratio).astype(np.int32)	    # np.arange(new_width) -> массив [0, 1, ..., new_width - 1]

	# Индексация с использованием массивов в numpy
	# img[[0, 2, 4], [0, 2, 4]] -> выборка пикселей с координатами (0,0), (0,2), (0,4), (2,0), (2,2), (2,4), (4,0), (4,2), (4,4)
	res_img = img[y_indices[:, None], x_indices]

	return res_img


def image_sepia(img):
	"""
	Функция применения фотоэффекта сепии к изображению

	:param img: Исходное изображение
	:return: Изменённое изображение с эффектом сепии
	"""

	weights = np.array([[0.393, 0.769, 0.189],
	                    [0.349, 0.686, 0.168],
	                    [0.272, 0.534, 0.131]][::-1])   # Параметры инвертированы, т.к. opencv хранит изображения в BGR

	res_img = np.dot(img[:,  :, :3], weights.T)
	res_img = np.clip(res_img, 0, 255).astype(np.uint8)

	return res_img


def image_vignette_rad(img, rad):
	"""
	Функция применения фотоэффекта виньетки к изображению

	:param img: Исходное изображение
	:param rad: Радиус виньетки (в процентах от всего изображения)
	:return: Изменённое изображение с эффектом виньетки
	:raise ValueError("Invalid radius"): rad <= 0
	"""

	if rad <= 0:
		raise ValueError("Invalid radius.")

	# Размеры изображения и центр
	rows, cols = img.shape[:2]
	center_x, center_y = cols / 2, rows / 2

	# Радиус виньетки
	max_distance = np.sqrt(center_x**2 + center_y**2)
	radius = (rad / 100) * max_distance

	# Расстояние до центра в диапазоне [0, 1] с учётом радиуса
	img_Y, img_X = np.ogrid[:rows, :cols]							# задаёт матрицы размера (rows, 1) || (1, cols) и значения 1...val

	# Создаём маску для изменения значений пикселей
	distance_from_center = np.sqrt((img_X - center_x) ** 2 + (img_Y - center_y) ** 2)		# X + Y -> матрица размера img_cv
	mask = np.clip(1 - (distance_from_center / radius), 0, 1)					# значения больше radius зануляются

	res_img = (img * mask[..., np.newaxis]).astype(np.uint8)

	return res_img


def image_pixelate(img, win_name, pixel_size):
	"""
	Функция пикселизации выделенной прямоугольной области изображения

	:param img: Исходное изображение
	:param win_name: Имя окна с изображением
	:param pixel_size: Размер пикселизации
	:return: Изменённое изображение с эффектом пикселизации выделенной прямоугольной области
	:raise ValueError("Invalid selected area"): Выделенная область слишком мала
	"""

	if pixel_size <= 0:
		raise ValueError("Invalid pixel size.")

	# Обрезаем изображение
	coords = cv.selectROI(win_name, img, showCrosshair=False, printNotice=False)
	cropped_image = img[coords[1]:coords[1] + coords[3],
					coords[0]:coords[0] + coords[2]]

	# Получаем размеры изображения после разделения на блоки (блок - отдельный пиксель)
	new_h = (cropped_image.shape[0] // pixel_size) * pixel_size
	new_w = (cropped_image.shape[1] // pixel_size) * pixel_size

	if new_h == 0 or new_w == 0:
		raise ValueError("Invalid selected area.")

	# Разбиваем исходное изображение на блоки размером [pixel_size, pixel_size]
	# list [блоки по верт, pixel_size, блоки по гориз., pixel_size, 3]
	resize_image = cropped_image[:new_h, :new_w].reshape(new_h // pixel_size, pixel_size,
											new_w // pixel_size, pixel_size, cropped_image.shape[2])

	res_cropped_image = resize_image.mean(axis=(1, 3), dtype=int).repeat(pixel_size, axis=0).repeat(pixel_size, axis=1)
	res_cropped_image = np.clip(image_change_resolution(res_cropped_image, cropped_image.shape[1], cropped_image.shape[0]), 0, 255).astype(np.uint8)

	res_img = img
	res_img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]] = res_cropped_image
	return res_img


if __name__ == '__main__':
	args = cli_arguments()

	image_path = args.image_path
	input_image = cv.imread(image_path)

	if input_image is None:
		raise ValueError("Image not found or unable to load.")

	input_image = cv.resize(input_image, [int(x / 2) for x in input_image.shape[:2][::-1]])

	cv.imshow("Original", input_image)

	filter_type = args.filter_type
	match filter_type:
		case "grey":
			output_image = image_grey(input_image)
		case "resize":
			output_image = image_change_resolution(input_image, args.width, args.height)
		case "sepia":
			output_image = image_sepia(input_image)
		case "vignette":
			output_image = image_vignette_rad(input_image, args.radius)
		case "pixelate":
			output_image = image_pixelate(input_image, "Original", args.pixel_size)

	cv.imshow("Image", output_image)
	cv.waitKey(0)
