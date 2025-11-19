import argparse

from app import VehicleDetectionApp


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Практическая работа №2. Детектирование транспортных средств с OpenCV DNN"
    )
    parser.add_argument(
        "--images", "-i", type=str, required=True,
        help="Путь к директории с кадрами"
    )
    parser.add_argument(
        "--annotations", "-a", type=str, required=True,
        help="Путь к файлу с разметкой"
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        choices=["yolo", "ssd", "rcnn"],
        help="Модель детектирования: yolo | ssd | rcnn"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Порог уверенности"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Не показывать окна с визуализацией"
    )
    parser.add_argument(
        "--show-gt", action="store_true",
        help="Отрисовывать Ground Truth"
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    app = VehicleDetectionApp()
    app.run(
        images_dir=args.images,
        annotations_path=args.annotations,
        model_key=args.model,
        confidence_threshold=args.conf,
        display=not args.no_display,
        show_ground_truth=args.show_gt
    )


if __name__ == "__main__":
    main()
