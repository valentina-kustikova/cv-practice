import os
import sys
import cv2
import numpy as np
from cli import cli_argument_parser
from filters.base import ImageFilter


def read_image(image_path: str) -> np.ndarray:
    """Чтение изображения с обработкой исключений."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден по пути: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"Не удалось декодировать изображение. Файл поврежден или имеет неверный формат: {image_path}"
        )

    return image


def main():
    args = cli_argument_parser()

    try:
        image = read_image(args.input)

        filter_kwargs = {}
        if args.filter == "resize":
            filter_kwargs = {
                "target_width": args.width,
                "target_height": args.height,
            }
        elif args.filter == "fade":
            filter_kwargs = {"factor": args.factor}
        elif args.filter == "old_photo":
            filter_kwargs = {
                "noise_amount": args.noise,
                "scratch_count": args.scratches,
            }

        image_filter = ImageFilter.get_filter(args.filter, **filter_kwargs)
        result_image = image_filter.apply_filter(image)

        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        success = cv2.imwrite(args.output, result_image)
        if not success:
            raise IOError(f"Не удалось сохранить изображение в {args.output}")

        print(
            f"Фильтр '{args.filter}' успешно применен. Результат сохранен в '{args.output}'."
        )

    except Exception as e:
        print(f"Ошибка при выполнении: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()