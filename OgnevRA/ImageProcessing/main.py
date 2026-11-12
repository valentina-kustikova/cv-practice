import cv2
from Filters import (resize_image, gray, antique, fade, infrared, negative, matte, old, neon, choose_image,
                     choose_filter)
def main():
    img = choose_image() #Выбираем изображение
    if img is None:
        print("Не удалось прочитать изображение")
        return
    selected = choose_filter() #Выбираем фильтр
    if selected is None:
        return
    name, filter_function = selected
    res=filter_function(img)
    h, w =res.shape[:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL ) #Название окна
    cv2.resizeWindow(name, int(w*700/h), 700 ) #Размер окна
    cv2.imshow(name, res) #Показ результата
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()