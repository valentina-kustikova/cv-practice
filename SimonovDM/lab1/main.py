from script import *
import sys

def main():
    args = parser()
    img = read_image(args.image)
    try:
        args_dict = vars(args).copy()
        args_dict.pop("image", None)
        args_dict.pop("filter_type", None)

        filter_obj = Filter.create_filter(args.filter_type, **args_dict)
        result = filter_obj.apply_filter(img)
        if result is not None:
            cv2.imshow("Original", img)
            cv2.imshow("Result", result)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()
    except ValueError as e:
        logging.error(f"Ошибка: {e}")
    except FileNotFoundError as e:
        logging.error(f"Файл не найден: {e}")
    except Exception as e:
        logging.error(f"Непредвиденная ошибка: {e}")

if __name__ == "__main__":
    sys.exit(main() or 0)