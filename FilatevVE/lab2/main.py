from lib.common import *

def vis(dets, srcimg, letterbox_scale, model_type='yolox'):
    """Универсальная функция визуализации для всех моделей"""
    res_img = srcimg.copy()

    for det in dets:
        bbox = det[:4]
        score = det[-2]
        cls_id = int(det[-1])

        x0, y0, x1, y1 = unletterbox(bbox, res_img.shape[:2], letterbox_scale, model_type)

        class_name = classes[cls_id]

        label = "{:s}: {:.3f}".format(class_name, score)

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(res_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.rectangle(res_img, (x0, y0 - text_height - baseline),
                      (x0 + text_width, y0), (0, 255, 0), -1)

        cv2.putText(res_img, label, (x0, y0 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return res_img


def get_model_choice():
    """Получает выбор модели от пользователя"""
    print("\nВыберите модель:")
    print("1 - NanoDet (быстрая)")
    print("2 - YoloX (точная)")
    print("3 - SSD (сбалансированная)")

    model_choice = input("Введите номер модели (1-3): ").strip()
    model_map = {"1": "nanodet", "2": "yolox", "3": "ssd"}
    return model_map.get(model_choice, "yolox")


def initialize_model(model_name):
    """Инициализирует модель детекции"""
    config = MODEL_CONFIGS[model_name]
    model_path = f"models/{model_name}.onnx"

    model_params = config['params'].copy()
    model_params.update({
        'backendId': cv2.dnn.DNN_BACKEND_OPENCV,
        'targetId': cv2.dnn.DNN_TARGET_CPU
    })

    return config['class'](modelPath=model_path, **model_params), config


def process_and_show_image(model, config, model_name, image_path, window_name):
    """Обрабатывает и показывает одно изображение"""
    image = resize_image_proportional(cv2.imread(image_path), 900, 1300)

    input_blob = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_blob, letterbox_scale = letterbox(input_blob, config['input_size'], model_name)

    preds = model.infer(input_blob)

    result_img = vis(preds, image, letterbox_scale, model_name)

    height, width = result_img.shape[:2]
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, result_img)
    return True

def process_and_show_video(model, config, model_name, video_path, window_name):
    """Обрабатывает и показывает одно видео"""

    video = cv2.VideoCapture(video_path)
    try:
        while True:
            success, image = video.read()
            if not success:
                break
            
            image = resize_image_proportional(image, 900, 1300)

            input_blob = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_blob, letterbox_scale = letterbox(input_blob, config['input_size'], model_name)

            preds = model.infer(input_blob)

            result_img = vis(preds, image, letterbox_scale, model_name)

            height, width = result_img.shape[:2]
            cv2.resizeWindow(window_name, width, height)
            cv2.imshow(window_name, result_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        video.release()

    return True


def handle_key_navigation(key, current_idx, images_length):
    """Обрабатывает навигационные клавиши и возвращает новый индекс"""
    if key in [ord('d'), ord('в')]:  # D - следующее изображение
        return (current_idx + 1) % images_length
    elif key in [ord('a'), ord('ф')]:  # A - предыдущее изображение
        return (current_idx - 1) % images_length
    return current_idx


def run_image_navigation(image_index, model_name, images):
    """Запуск навигации по изображениям с клавишами D и A"""
    model, config = initialize_model(model_name)
    current_idx = image_index
    window_name = f"Image Navigation - {model_name.upper()}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    image_path = os.path.join("images", images[current_idx])

    process_and_show_image(model, config, model_name, image_path, window_name)

    print("\nУправление:")
    print("D - следующее изображение")
    print("A - предыдущее изображение")
    print("ESC - выход")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key in [ord('d'), ord('в'), ord('a'), ord('ф')]:  # Навигация
            new_idx = handle_key_navigation(key, current_idx, len(images))
            if new_idx != current_idx:
                current_idx = new_idx
                image_path = os.path.join("images", images[current_idx])
                process_and_show_image(model, config, model_name, image_path, window_name)

    cv2.destroyAllWindows()


def handle_image_mode():
    """Основная функция для обработки режима изображений"""
    images, _ = load_images_from_folder("images")

    model_name = get_model_choice()

    run_image_navigation(0, model_name, images)


def handle_video_mode():
    """Основная функция для обработки режима видео"""
    videos, _ = load_videos_from_folder("videos")

    print("Доступны видео от 1 до", len(videos))

    videos_choice = input("Введите номер видео: ").strip()
    video_index = int(videos_choice) - 1

    model_name = get_model_choice()
    video_path = os.path.join("videos", videos[video_index])

    model, config = initialize_model(model_name)
    window_name = f"Image Navigation - {model_name.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    process_and_show_video(model, config, model_name, video_path, window_name)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    uploading_models()

    print("\nВыберите тип источника:")
    print("1 - Изображение")
    print("2 - Видео")

    source_type = input("Введите номер типа источника (1-2): ").strip()

    if source_type == "1":
        handle_image_mode()
    elif source_type == "2":
        handle_video_mode()
    else:
        print("Неверный выбор!")