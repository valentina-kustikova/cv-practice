from cli import cli_argument_parser
from image_io import read_image, save_image
from filters.base import ImageFilter


def main():
    args = cli_argument_parser()

    try:
        image = read_image(args.input)

        image_filter = ImageFilter.get_filter(args.filter)

        if args.filter == "resize":
            result = image_filter.apply(
                image,
                width=args.width,
                height=args.height,
                scale_x=args.scale_x,
                scale_y=args.scale_y,
            )

        elif args.filter == "fade":
            color = tuple(
                int(value.strip())
                for value in args.fade_color.split(",")
            )
            if len(color) != 3 or any(value < 0 or value > 255 for value in color):
                raise ValueError("Цвет должен состоять из трёх чисел от 0 до 255.")

            result = image_filter.apply(
                image,
                alpha=args.alpha,
                color=color,
            )

        elif args.filter == "film":
            result = image_filter.apply(
                image,
                strength=args.strength,
                noise=args.film_noise,
            )

        elif args.filter == "matte":
            result = image_filter.apply(
                image,
                radius=args.radius,
                strength=args.matte_strength,
                softness=args.softness,
            )

        elif args.filter == "noise":
            result = image_filter.apply(
                image,
                amount=args.noise,
                scratch_density=args.scratch_density,
            )

        elif args.filter == "neon":
            result = image_filter.apply(
                image,
                strength=args.neon_strength,
                blur=args.neon_blur,
            )

        else:
            result = image_filter.apply(image)

        save_image(result, args.output)
        print(f"Готово! Результат сохранён: {args.output}")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()

