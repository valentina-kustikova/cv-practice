import argparse
import os

import cv2

from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="путь к изображению")
    parser.add_argument("-f", "--filter", nargs="+", required=True, help="фильтры или all")
    parser.add_argument("-p", "--param", nargs="*", default=[], help="параметры ключ=значение")
    parser.add_argument("-o", "--output", default="results", help="папка для результатов")
    args = parser.parse_args()
    return args


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("не удалось открыть " + path)
    return image


def read_params(param_list):
    params = {}
    for text in param_list:
        if "=" not in text:
            print("Параметр " + text + " нужно писать так: ключ=значение")
            continue
        parts = text.split("=")
        params[parts[0]] = parts[1]
    return params


def main():
    args = cli_argument_parser()

    try:
        image = read_image(args.input)
    except FileNotFoundError as error:
        print("Ошибка:", error)
        return

    params = read_params(args.param)

    filter_names = args.filter
    if filter_names[0] == "all":
        filter_names = ImageFilter.names()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    image_name = os.path.basename(args.input)
    image_name = image_name.split(".")[0]

    for filter_name in filter_names:
        try:
            image_filter = ImageFilter.get_filter(filter_name, **params)
            result = image_filter.apply_filter(image)
        except ValueError as error:
            print("Ошибка:", error)
            continue

        save_path = args.output + "/" + image_name + "_" + filter_name + ".jpg"
        cv2.imwrite(save_path, result)
        print("Сохранено:", save_path)


main()