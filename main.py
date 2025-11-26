import cv2
import numpy as np

glare = np.zeros((200, 200, 4), dtype=np.uint8)
cv2.circle(glare, (100, 100), 30, (255, 255, 255, 255), -1)
cv2.circle(glare, (100, 100), 100, (255, 220, 180, 150), -1)
cv2.imwrite('glare.png', glare)

watercolor = np.full((600, 800, 4), (245, 240, 230, 60), dtype=np.uint8)
for _ in range(120):
    x = np.random.randint(0, 800)
    y = np.random.randint(0, 600)
    r = np.random.randint(20, 60)
    gray = np.random.randint(160, 200)
    alpha = np.random.randint(80, 150) 
    cv2.circle(watercolor, (x, y), r, (gray, gray, gray, alpha), -1)
cv2.imwrite('watercolor_paper.png', watercolor)

print("✅ Текстуры созданы")

from google.colab import files
print("Загрузите изображение:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
image = cv2.imread(filename)
if image is None:
    raise Exception("Изображение не загружено")

original = image.copy()
h, w = original.shape[:2]

from filters import *
import matplotlib.pyplot as plt

resized = resize_image(original.copy(), width=400)
sepia = apply_sepia(original.copy())
vignette = apply_vignette(original.copy(), strength=0.7)
pixelized = pixelize_region(original.copy(), w//4, h//4, 3*w//4, 3*h//4, pixel_size=15)
solid_border = apply_solid_border(original.copy(), border_width=20, color=(0, 0, 255))
custom_border = apply_custom_border(original.copy(), border_width=8, color=(0, 255, 0))
flare = apply_lens_flare(original.copy(), 'glare.png', flare_position=(w//5, h//5))
watercolor = apply_watercolor_texture(original.copy(), 'watercolor_paper.png')

images = [
    (original, "Оригинал"),
    (resized, "Разрешение"),
    (sepia, "Сепия"),
    (vignette, "Виньетка"),
    (pixelized, "Пикселизация"),
    (solid_border, "Сплошная рамка"),
    (custom_border, "Фигурная рамка"),
    (flare, "Блик"),
    (watercolor, "Акварель")
]

plt.figure(figsize=(18, 10))
for i, (img, title) in enumerate(images):
    plt.subplot(3, 3, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=11)
    plt.axis('off')
plt.tight_layout()
plt.show()

print("✅ Готово!")
