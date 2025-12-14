import os
import subprocess
import cv2
import filters

def ask_str(prompt: str, default: str | None = None) -> str:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        if s != "":
            return s
        print("Пустой ввод. Попробуйте ещё раз.")


def ask_int(prompt: str, default: int, min_v: int | None = None, max_v: int | None = None) -> int:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            v = int(default)
        else:
            try:
                v = int(s)
            except ValueError:
                print("Нужно целое число.")
                continue

        if min_v is not None and v < min_v:
            print(f"Значение должно быть >= {min_v}")
            continue
        if max_v is not None and v > max_v:
            print(f"Значение должно быть <= {max_v}")
            continue
        return v


def ask_bool(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        s = input(f"{prompt} ({hint}): ").strip().lower()
        if s == "":
            return default
        if s in ("y", "yes", "1", "да", "д"):
            return True
        if s in ("n", "no", "0", "нет", "н"):
            return False
        print("Введите Y или N.")


def ask_color_bgr(prompt: str, default=(0, 0, 0)):
    while True:
        s = input(f"{prompt} B,G,R [{default[0]},{default[1]},{default[2]}]: ").strip()
        if s == "":
            return default
        parts = s.split(",")
        if len(parts) != 3:
            print("Формат: B,G,R (например 0,0,0).")
            continue
        try:
            vals = [int(p.strip()) for p in parts]
        except ValueError:
            print("Цвет должен содержать целые числа.")
            continue
        vals = [max(0, min(255, v)) for v in vals]
        return tuple(vals)


def choose_from(prompt: str, options: list[str], default: str):
    opts = set(options)
    while True:
        s = input(f"{prompt} [{default}]: ").strip().lower()
        if s == "":
            return default
        if s in opts:
            return s
        print(f"Неверно. Допустимо: {', '.join(options)}")

def build_out_path(inp: str, suffix: str) -> str:
    base, ext = os.path.splitext(inp)
    if ext == "":
        ext = ".png"
    return f"{base}_{suffix}{ext}"


def load_image_interactive() -> tuple[str, any]:
    while True:
        path = ask_str("Введите путь к изображению: ")
        if not os.path.exists(path):
            print("Файл не найден. Проверьте путь и расширение (.jpg/.png и т.п.).")
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print("OpenCV не смог прочитать файл (возможно, формат/файл битый).")
            continue
        return path, img


def open_with_viewer(path: str) -> bool:
    try:
        subprocess.Popen(
            ["xdg-open", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"[Не удалось открыть просмотрщик] {e}")
        return False

MENU = [
    ("resolution", "Изменение разрешения (change_resolution)"),
    ("sepia",      "Фотоэффект сепии (apply_sepia)"),
    ("vignette",   "Фотоэффект виньетки (apply_vignette)"),
    ("pixelate",   "Пикселизация ROI (pixelate_region)"),
    ("rect_frame", "Прямоугольная рамка (add_rect_frame)"),
    ("shape_frame","Фигурная рамка (add_shape_frame)"),
    ("lens_flare", "Блики объектива (add_lens_flare)"),
    ("watercolor", "Текстура бумаги (add_watercolor_texture)"),
]


def choose_filter() -> str:
    print("\nВыберите фильтр:")
    for i, (_, title) in enumerate(MENU, start=1):
        print(f"  {i}. {title}")
    print("  0. Выход")

    n = ask_int("Номер", 1, min_v=0, max_v=len(MENU))
    if n == 0:
        return "exit"
    return MENU[n - 1][0]


def apply_filter(img, key: str):
    H, W = img.shape[:2]

    if key == "resolution":
        new_w = ask_int("Новая ширина", 800, min_v=1)
        new_h = ask_int("Новая высота", 600, min_v=1)
        out = filters.change_resolution(img, new_w, new_h)
        return out, f"resolution_{new_w}x{new_h}"

    if key == "sepia":
        out = filters.apply_sepia(img)
        return out, "sepia"

    if key == "vignette":
        sigma = ask_int("Sigma (меньше = сильнее затемнение краёв)", 200, min_v=1)
        out = filters.apply_vignette(img, sigma=sigma)
        return out, f"vignette_sigma{sigma}"

    if key == "pixelate":
        print(f"Размер изображения: {W}x{H}")
        x = ask_int("ROI x", 0, min_v=0, max_v=W-1)
        y = ask_int("ROI y", 0, min_v=0, max_v=H-1)
        rw = ask_int("ROI width", min(300, W), min_v=1, max_v=W)
        rh = ask_int("ROI height", min(200, H), min_v=1, max_v=H)
        block = ask_int("Размер блока (block_size)", 10, min_v=1, max_v=300)
        out = filters.pixelate_region(img, x, y, rw, rh, block)
        return out, f"pixelate_{x}_{y}_{rw}x{rh}_b{block}"

    if key == "rect_frame":
        fw = ask_int("Толщина рамки (frame_width)", 20, min_v=1, max_v=max(1, min(H, W)//2))
        color = ask_color_bgr("Цвет рамки", (0, 0, 0))
        out = filters.add_rect_frame(img, fw, color)
        return out, f"rectframe_{fw}_{color[0]}_{color[1]}_{color[2]}"

    if key == "shape_frame":
        shape = choose_from("Форма (ellipse/circle/diamond/star)", ["ellipse", "circle", "diamond", "star"], "ellipse")
        color = ask_color_bgr("Цвет рамки", (0, 0, 0))
        out = filters.add_shape_frame(img, shape=shape, frame_color=color)
        return out, f"shapeframe_{shape}_{color[0]}_{color[1]}_{color[2]}"

    if key == "lens_flare":
        out = filters.add_lens_flare(img)
        return out, "lensflare"

    if key == "watercolor":
        out = filters.add_watercolor_texture(img)
        return out, "watercolor"

    raise ValueError("Unknown filter key")


def main():
    print("Практическая работа №1 — интерактивное меню фильтров (железный просмотр)\n")

    img_path, img = load_image_interactive()

    while True:
        key = choose_filter()
        if key == "exit":
            print("Выход.")
            return

        try:
            out, suffix = apply_filter(img, key)
        except Exception as e:
            print(f"[Ошибка] {e}")
            continue

        out_path = build_out_path(img_path, suffix)
        if not cv2.imwrite(out_path, out):
            print(f"[Ошибка] Не удалось сохранить файл: {out_path}")
            continue

        print(f"[OK] Сохранено: {out_path}")

        if ask_bool("Открыть результат в системном просмотрщике (xdg-open)?", True):
            if open_with_viewer(out_path):
                print("[OK] Открыл (или попытался открыть) файл в просмотрщике.")
            else:
                print("[!] Не удалось запустить просмотрщик. Откройте файл вручную.")

        if not ask_bool("Применить ещё один фильтр к исходному изображению?", True):
            print("Готово.")
            return


if __name__ == "__main__":
    main()
