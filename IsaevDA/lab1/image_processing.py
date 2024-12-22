import argparse
import sys
import cv2 as cv
import os
import numpy as np

rect_start = (0, 0)
rect_end = (0, 0)
drawing = False

def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('-m', '--mode', help='Choose mode (image/imgproc)', dest='mode', default='image')
  parser.add_argument('-f', '--func', choices=['grayscale', 'resize', 'sepia', 'vignette', 'pixelation'],dest='func')
  parser.add_argument('-i', '--image',type=str,help='Path to input image', dest='image_path')
  parser.add_argument('-o', '--output', type=str, help='Output image name', dest='output_image',default='image_out.jpg')

  parser.add_argument('-v', '--vignette-intensity', type=float, help='Intensity of vignette 0 to 1 (not inclusive)', dest='vignette_intensity', default=0.15)
  parser.add_argument('-r', '--vignette-radius', type=float, help='Radius of vignette circle (0 to 1, where 1 is max)', dest='vignette_radius', default=0.5)
  parser.add_argument('-s', '--scale', type=float, dest='scale', default=0.5)
  parser.add_argument('-p', '--pixel-size', type=int, help='Block size for pixelation effect', dest='pixel_size', default=30)

  args = parser.parse_args()
  return args


def highgui_samples(image_path):
  img = read_image(image_path)

  cv.imshow('Init image', img)
  while True:
    key = cv.waitKey(0)
    if key == 27:  # Код клавиши Esc
      print("Escape pressed, closing window.")
      break

  cv.destroyAllWindows()


def generate_unique_filename(filename):
  base, ext = os.path.splitext(filename)
  count = 1

  while os.path.exists(filename):
    filename = f"{base}_{count}{ext}"
    count += 1
    
  return filename


def read_image(image_path):
  img = cv.imread(image_path)

  if img is None:
    print("Could not open or find the image")
    return
  
  return img


def show_end_image(img, output_image):
  cv.imshow('Processed Image', img)
  output_image = generate_unique_filename(output_image)
  cv.imwrite(output_image, img)
  
  while True:
    key = cv.waitKey(0)
    if key == 27:  # Код клавиши Esc
      print("Escape pressed, closing window.")
      break
  
  cv.destroyAllWindows()

# Функция перевода изображения в оттенки серого
def grayscale(img):
  gray_img = np.zeros_like(img,np.uint8)

  gray_img[:,:,0] = 0.3*img[:,:,2] + 0.59*img[:,:,1] + 0.11*img[:,:,0]
  gray_img[:,:,1] = gray_img[:,:,0]
  gray_img[:,:,2] = gray_img[:,:,0]

  return gray_img

# Функция изменения разрешения изображения
def resize(img, scale):
  height, width, n = img.shape

  new_height = int(height * scale)
  new_width = int(width * scale)

  # Создаем сетки индексов для нового изображения
  row_indices = (np.arange(new_height) / scale).astype(int)
  col_indices = (np.arange(new_width) / scale).astype(int)

  row_indices = np.clip(row_indices, 0, height - 1)
  col_indices = np.clip(col_indices, 0, width - 1)

  # Применяем индексы для создания нового изображения
  resize_img = img[row_indices[:, np.newaxis], col_indices]

  print(f'Original shape: {img.shape}')
  print(f'Resized shape: {resize_img.shape}')

  return resize_img

# Функция применения фотоэффекта сепии к изображению
def sepia(img):
  sepia_img = np.zeros_like(img, np.uint8)

  sepia_img[:,:,2] = np.clip(0.393*img[:,:,2] + 0.769*img[:,:,1] + 0.189*img[:,:,0], 0, 255)
  sepia_img[:,:,1] = np.clip(0.349*img[:,:,2] + 0.686*img[:,:,1] + 0.168*img[:,:,0], 0, 255)
  sepia_img[:,:,0] = np.clip(0.272*img[:,:,2] + 0.534*img[:,:,1] + 0.131*img[:,:,0], 0, 255)

  return sepia_img


def create_vignette_mask(height, width, intensity, radius):
  y, x = np.ogrid[:height, :width]
  center_x = width // 2
  center_y = height // 2

  max_radius = min(center_x, center_y) * radius
  distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

  mask = np.exp(-((distance / max_radius) ** 2) * intensity)
  mask = mask / np.max(mask)

  return mask

# Функция применения фотоэффекта виньетки к изображению
def vignette(img, vignette_intensity, vignette_radius):
  height, width, n = img.shape

  mask = create_vignette_mask(height, width, vignette_intensity, vignette_radius)

  img = img * mask[:, :, np.newaxis]
  img = img.astype(np.uint8)

  return img

# Функция пикселизации заданной прямоугольной области изображения
def pixelation(img, rect_start, rect_end, block_size):
  x1, y1 = rect_start
  x2, y2 = rect_end

  output = img.copy()

  for i in range(y1, y2, block_size):
    for j in range(x1, x2, block_size):
      block_y2 = min(i + block_size, y2)
      block_x2 = min(j + block_size, x2)
      block = output[i:block_y2, j:block_x2]
      avg_color = np.mean(block, axis=(0, 1), dtype=int)
      output[i:block_y2, j:block_x2] = avg_color

  return output

# Обработчик событий мыши для выбора области
def mouse_callback(event, x, y, flags, param):
  global rect_start, rect_end, drawing

  if event == cv.EVENT_LBUTTONDOWN:
    rect_start = (x, y)
    drawing = True

  elif event == cv.EVENT_MOUSEMOVE:
    if drawing:
      img_copy = param.copy()
      cv.rectangle(img_copy, rect_start, (x, y), (0, 255, 0), 2)
      cv.imshow('Image', img_copy)

  elif event == cv.EVENT_LBUTTONUP:
    rect_end = (x, y)
    drawing = False


def select_pixelation_area(img):
  global rect_start, rect_end

  cv.imshow('Image', img)
  cv.setMouseCallback('Image', mouse_callback, img)

  while True:
    key = cv.waitKey(10) & 0xFF
    if key == 13:  # Нажатие Enter для подтверждения
      break

  cv.destroyAllWindows()
  return rect_start, rect_end


def main():
  args = parse_arguments()
    
  if args.mode == 'image':
    highgui_samples(args.image_path)
  elif args.mode == 'imgproc':
    img = read_image(args.image_path)

    if args.func == 'grayscale':
      img = grayscale(img)
    elif args.func =='resize':
      if args.scale <= 0:
        print("Значение должно быть больше 0!")
        return
      img = resize(img, args.scale)
    elif args.func =='sepia':
      img = sepia(img)
    elif args.func == 'vignette':
      if args.vignette_intensity <= 0 or args.vignette_intensity >= 1:
        print("Интенсивность виньетки должна быть между 0 и 1 (не включительно)!")
        return
      if args.vignette_radius <= 0 or args.vignette_radius > 1:
        print("Радиус виньетки должен быть между 0 и 1!")
        return
      img = vignette(img, args.vignette_intensity, args.vignette_radius)
    elif args.func == 'pixelation':
      if args.pixel_size <= 1:
        print("Значение должно быть больше 1!")
        return
      rect_start, rect_end = select_pixelation_area(img)
      img = pixelation(img, rect_start, rect_end, args.pixel_size)

    show_end_image(img, args.output_image)
  else:
    raise 'Unsupported \'mode\' value'


if __name__ == '__main__':
  sys.exit(main() or 0)