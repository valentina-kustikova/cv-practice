import cv2
import argparse
from filters import *


def main():
    parser = argparse.ArgumentParser(
        description="Применение фильтров к изображению"
    )

    parser.add_argument("image", help="Путь к изображению для обработки")
    parser.add_argument("filter",
                        help="Тип фильтра (resize, sepia, vignette, pixelate, frame, figframe, flare, texture)")
    parser.add_argument("params", nargs="*", help="Дополнительные параметры фильтра (опционально)")

    args = parser.parse_args()
    img = cv2.imread(args.image)

    if img is None:
        print("Ошибка: не удалось загрузить изображение.")
        return

    ftype = args.filter.lower()
    params = args.params
    result = None

    # Обработка выбранного фильтра
    if ftype == "resize":
        scale = float(params[0]) if params else 0.5
        result = resize_image(img, scale)

    elif ftype == "sepia":
        result = apply_sepia(img)

    elif ftype == "vignette":
        strength = float(params[0]) if params else 0.5
        result = apply_vignette(img, strength)

    elif ftype == "pixelate":
        if len(params) < 4:
            print("Ошибка: для pixelate нужны параметры x1 y1 x2 y2 [pixel_size]")
            return
        x1, y1, x2, y2 = map(int, params[:4])
        pixel_size = int(params[4]) if len(params) > 4 else 10
        result = pixelate_region(img, x1, y1, x2, y2, pixel_size)

    elif ftype == "frame":
        color = (0, 0, 255)
        thickness = int(params[0]) if params else 20
        result = add_rectangular_frame(img, color, thickness)

    elif ftype == "figframe":
        color = (0, 255, 0)
        thickness = int(params[0]) if params else 20
        frame_type = params[1] if len(params) > 1 else "wave"
        result = add_figure_frame(img, color, thickness, frame_type)

    elif ftype == "flare":
        result = add_lens_flare(img)

    elif ftype == "texture":
        scale = int(params[0]) if params else 11
        intensity = float(params[1]) if len(params) > 1 else 0.2
        result = add_paper_texture(img, scale, intensity)

    else:
        print("Ошибка: неизвестный тип фильтра.")
        return

    # Отображение изображений
    cv2.imshow("Original", img)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
