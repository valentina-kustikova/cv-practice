# script.py
import cv2
import numpy as np
import argparse
import sys
import os
from filters import *

def main():
    parser = argparse.ArgumentParser(description='Практическая работа №1: фильтры изображений')
    parser.add_argument('-i', '--image', required=True, help='Путь к изображению')
    parser.add_argument('-f', '--filter', required=True, help='Тип фильтра: resize, sepia, vignette, pixelate, frame, decorative, flare, watercolor')
    parser.add_argument('-o', '--output', default='output.jpg', help='Путь к выходному файлу')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--strength', type=float, default=0.6)
    parser.add_argument('--x', type=int, default=100)
    parser.add_argument('--y', type=int, default=100)
    parser.add_argument('--pixel_size', type=int, default=10)
    parser.add_argument('--frame_width', type=int, default=20)
    parser.add_argument('--border_type', type=str, default='dashed')
    parser.add_argument('--flare_x', type=int, default=100)
    parser.add_argument('--flare_y', type=int, default=100)
    
    args = parser.parse_args()
    
    # Загрузка изображения
    image = cv2.imread(args.image)
    if image is None:
        print(f"Ошибка: не удалось загрузить {args.image}")
        sys.exit(1)
    
    print(f"Изображение загружено: {image.shape}")
    
    # Применение фильтра
    if args.filter == 'resize':
        result = resize_image(image, args.width, args.height)
    elif args.filter == 'sepia':
        result = apply_sepia(image)
    elif args.filter == 'vignette':
        result = apply_vignette(image, args.strength)
    elif args.filter == 'pixelate':
        # Пикселизация области 200x200 от (x,y)
        h, w = image.shape[:2]
        x2 = min(args.x + 200, w)
        y2 = min(args.y + 200, h)
        result = pixelize_region(image, args.x, args.y, x2 - args.x, y2 - args.y, args.pixel_size)
    elif args.filter == 'frame':
        result = apply_solid_border(image, args.frame_width)
    elif args.filter == 'decorative':
        result = apply_custom_border(image, args.frame_width, args.border_type)
    elif args.filter == 'flare':
        # Для локального запуска — генерируем текстуру блика
        glare = np.zeros((200, 200, 4), dtype=np.uint8)
        cv2.circle(glare, (100, 100), 50, (255, 255, 255, 200), -1)
        cv2.imwrite('glare.png', glare)
        result = apply_lens_flare(image, 'glare.png', (args.flare_x, args.flare_y))
    elif args.filter == 'watercolor':
        # Генерируем текстуру акварели
        h, w = image.shape[:2]
        wc = np.full((h, w, 4), (240, 235, 225, 40), dtype=np.uint8)
        for _ in range(100):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            r = np.random.randint(10, 30)
            cv2.circle(wc, (x, y), r, (180, 170, 160, 80), -1)
        cv2.imwrite('watercolor_paper.png', wc)
        result = apply_watercolor_texture(image, 'watercolor_paper.png')
    else:
        print(f"Неизвестный фильтр: {args.filter}")
        sys.exit(1)
    
    # Сохранение и отображение
    cv2.imwrite(args.output, result)
    print(f"Результат сохранён в {args.output}")
    
    # Показываем окно (можно закрыть)
    cv2.imshow('Исходное', image)
    cv2.imshow('Результат', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()