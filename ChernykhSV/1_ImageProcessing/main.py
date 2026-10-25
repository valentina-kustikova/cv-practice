
import sys
import os # путь к файлу
import argparse # для парсинга аргументов командной строки
import cv2 as cv

import filter

import argparse




def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="Путь к исходному изображению")
    parser.add_argument("-o", "--output", required=True, help="Путь для сохранения результата")
    parser.add_argument("-f", "--flag", required=True, choices=["resize", "grayscale", "antique", "fade", "infrared", "matte", "aged", "neon"], help="Тип фильтра")


    parser.add_argument("--scale", type=float, default=0.5, help="Коэффициент масштаба для resize (default: 0.5)")
    parser.add_argument("--fade_level", type=float, default=150, help="Уровень серого для fade (default: 150)")
    parser.add_argument("--sepia", type=float, default=0.7, help="Сила эффекта сепии для антиквариата 0..1 (default: 0.7)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Коэффициент выцветания изображения 0..1 (default: 0.6)")
    parser.add_argument("--infrared_strength", type=float, default=0.6, help="Коэффициент инфракрасной пленки (default: 0.6)")
    parser.add_argument("--matte_radius", type=float, default=0.8, help="Радиус маски (default: 0.8)")
    parser.add_argument("--matte_softness", type=float, default=0.2, help="Ширина зоны растушевки (default: 0.2)")
    parser.add_argument("--noise_amount", type=float, default=0.05, help="Уровень шума для scratches (default: 0.05)")
    parser.add_argument("--scratches_count", type=int, default=7, help="Число царапин для scratches (default: 7)")
    parser.add_argument("--neon_gain", type=int, default=1, help="Усиление контуров для неона (default: 1)")



    args = parser.parse_args()

    params = {
        "alpha":args.alpha,
        "scale": args.scale,
        "sepia": args.sepia,
        "noise_amount": args.noise_amount,
        "scratches_count": args.scratches_count,
        "fade_level": args.fade_level,
        "infrared-strength": args.infrared_strength,
        "matte_radius": args.matte_radius,
        "matte_softness": args.matte_softness,
        "neon_gain": args.neon_gain
    }

    return args.input, args.output, args.flag, params


def read_image(input_path):
    if not input_path:
        raise ValueError("Путь пустой")
    image = cv.imread(input_path) # функция для чтения изображения из файла, возвращает None если не удалось прочитать
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать файл: {input_path}")
    return image

    
def save_image(output_path, img):
    path, image = output_path, img
    
    if not cv.imwrite(path,image): # функция для сохранения изображения в файл, возвращает False если не удалось сохранить
        raise FileNotFoundError(f"Не удалось прочитать файл: {output_path}")
    

    



def show_image(winname: str, image) -> None: # функция для отображения изображения в отдельном окне, winname - название окна, image - изображение
    cv.imshow(winname, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def open_image(path_to_open_file):
    os.startfile(path_to_open_file)


def main():
    input_path, output_path, flag_filter, params = cli_argument_parser()
    image = read_image(input_path)
    f = filter.ImageFilter.get_filter(flag_filter, **params)
    result = f.apply_filter(image)
    save_image(output_path, result)
    show_image("result", result)


    return 0




if __name__=="__main__":
    sys.exit(main())

