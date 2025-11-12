import cv2
from filters import *
#Вставить то что ниже в консоль
#.venv\Scripts\activate
# Загружаем исходное изображение
image = cv2.imread("example.jpg")
flare_src = 'source/lens_flare.jpg'
paper_src = 'source/watercolor_paper.jpg' 
texture_src = 'source/texture_frame.jpg'

if image is None:
    raise FileNotFoundError("Файл example.jpg не найден.")

# 1. Изменение разрешения
resized = resize_image(image, 0.8)

# 2. Сепия
sepia = apply_sepia(resized)

# 3. Виньетка
vignette = apply_vignette(resized, strength=0.6)

# 4. Пикселизация области
#pixelated = pixelate_region(resized, 100, 100, 200, 200, pixel_size=15)

# 5. Прямоугольная рамка
rect_frame = add_rect_frame(resized, color=(0, 0, 255), thickness=25)

# 6. Фигурная рамка
shape_frame = add_shape_frame(resized, texture_src, thickness=50 )

# 7. Блики объектива
lens_flare = add_lens_flare(resized, flare_src, intensity=0.5)

# 8. Текстура бумаги
paper_texture = add_paper_texture(resized, texture_src, intensity=0.3)

# Отображение результатов
cv2.imshow("Original", image)
#cv2.imshow("Resized", resized)
cv2.imshow("Sepia", sepia)
cv2.imshow("Vignette", vignette)
#cv2.imshow("Pixelated", pixelated)
cv2.imshow("Rect Frame", rect_frame)
cv2.imshow("Shape Frame", shape_frame)
cv2.imshow("Lens Flare", lens_flare)
cv2.imshow("Paper Texture", paper_texture)

cv2.waitKey(0)
cv2.destroyAllWindows()
