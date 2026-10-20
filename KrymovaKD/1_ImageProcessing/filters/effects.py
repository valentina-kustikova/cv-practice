
import numpy as np

from .base import ImageFilter


class Matte(ImageFilter):
    """Создаёт белую мягкую овальную рамку по краям."""

    def apply(self, image, radius=0.58, strength=1.0, softness=2.2, **kwargs):
        if not 0 < radius < 1:
            raise ValueError("radius должен быть между 0 и 1")
        if not 0 <= strength <= 1:
            raise ValueError("strength должен быть от 0 до 1")
        if softness <= 0:
            raise ValueError("softness должен быть положительным")

        height, width = image.shape[:2]

        y, x = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
        distance = np.sqrt(x**2 + y**2)

        t = np.clip((distance - radius) / (1 - radius), 0, 1)
        mask = (t**softness * strength)[:, :, None]

        result = image.astype(np.float32) * (1 - mask) + 255 * mask

        return np.clip(result, 0, 255).astype(np.uint8)




class Noise(ImageFilter):
    """Состаривание фотографии: зерно, пыль, царапины и яркая засветка."""

    def apply(self, image, amount=0.3, scratch_density=0.1, **kwargs):
        if amount < 0:
            raise ValueError("amount не может быть отрицательным")
        if not 0 <= scratch_density <= 1:
            raise ValueError("scratch_density должен быть от 0 до 1")

        rng = np.random.default_rng()
        result = image.astype(np.float32).copy()
        height, width = image.shape[:2]

        # 1. Немного затемняем фотографию.
        result *= 0.88

        # 2. Добавляем плёночное зерно.
        grain = rng.normal(0, amount * 24, (height, width, 1))
        result += grain.astype(np.float32)

        # 3. Создаём тонкие царапины.
        light_scratches = np.zeros((height, width), dtype=np.float32)
        dark_scratches = np.zeros((height, width), dtype=np.float32)

        area = height * width / 1_000_000
        count = int(area * (500 + 1600 * scratch_density))

        directions = [
            rng.uniform(-0.18, 0.18),
            rng.uniform(1.35, 1.80),
            rng.uniform(-0.80, -0.45),
            rng.uniform(0.45, 0.80),
        ]

        for _ in range(count):
            x0 = int(rng.integers(0, width))
            y0 = int(rng.integers(0, height))

            kind = rng.choice(
                ["short", "medium", "long"],
                p=[0.70, 0.22, 0.08]
            )

            if kind == "short":
                length = int(rng.integers(3, 12))
            elif kind == "medium":
                length = int(rng.integers(12, 35))
            else:
                length = int(rng.integers(
                    30, max(31, min(110, max(height, width) // 2))
                ))

            angle = float(rng.choice(directions) + rng.normal(0, 0.14))
            curve = rng.uniform(-2.0, 2.0)
            waves = rng.uniform(0.5, 1.8)
            phase = rng.uniform(0, 2 * np.pi)
            drift = rng.normal(0, 0.10)

            steps = max(length * 3, 10)
            mask = (
                light_scratches
                if rng.random() < 0.78
                else dark_scratches
            )
            brightness = rng.uniform(0.20, 0.65)
            previous_x, previous_y = None, None

            for i in range(steps):
                t = i / max(steps - 1, 1)

                bend = (
                    curve * np.sin(np.pi * t)
                    + 0.55 * np.sin(t * waves * 2 * np.pi + phase)
                    + drift * length * t
                )

                px = int(round(
                    x0 + length * t * np.cos(angle)
                    - bend * np.sin(angle)
                ))
                py = int(round(
                    y0 + length * t * np.sin(angle)
                    + bend * np.cos(angle)
                ))

                if not (0 <= px < width and 0 <= py < height):
                    previous_x, previous_y = None, None
                    continue

                if rng.random() < 0.055:
                    previous_x, previous_y = None, None
                    continue

                if previous_x is not None:
                    distance = max(
                        abs(px - previous_x),
                        abs(py - previous_y),
                        1
                    )

                    xs = np.rint(np.linspace(
                        previous_x, px, distance + 1
                    )).astype(int)
                    ys = np.rint(np.linspace(
                        previous_y, py, distance + 1
                    )).astype(int)

                    for j, (lx, ly) in enumerate(zip(xs, ys)):
                        if 0 <= lx < width and 0 <= ly < height:
                            local_brightness = brightness * (
                                0.78 + 0.22 * np.sin(
                                    np.pi * j / max(len(xs) - 1, 1)
                                )
                            )
                            mask[ly, lx] = max(
                                mask[ly, lx], local_brightness
                            )

                mask[py, px] = max(mask[py, px], brightness)
                previous_x, previous_y = px, py

        # Смягчаем царапины.
        light_soft = box_blur(light_scratches)
        dark_soft = box_blur(dark_scratches)

        light = light_scratches * 0.72 + light_soft * 0.28
        dark = dark_scratches * 0.70 + dark_soft * 0.30

        texture = rng.uniform(
            0.82, 1.08, size=(height, width)
        ).astype(np.float32)
        light *= texture

        # Светлые тёплые царапины.
        result[:, :, 0] += light * 58
        result[:, :, 1] += light * 64
        result[:, :, 2] += light * 68

        # Тёмные царапины.
        result -= dark[:, :, None] * 28

        # 4. Добавляем пыль.
        dust_count = int(height * width * 0.00022)

        for _ in range(dust_count):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            radius = int(rng.choice([1, 1, 1, 1, 2, 3]))
            brightness = float(rng.uniform(15, 75))

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    px, py = x + dx, y + dy

                    if not (0 <= px < width and 0 <= py < height):
                        continue

                    distance = np.sqrt(dx * dx + dy * dy)

                    if distance <= radius:
                        value = brightness * (
                            1 - distance / (radius + 1)
                        )
                        result[py, px, 0] += value * 0.82
                        result[py, px, 1] += value * 0.92
                        result[py, px, 2] += value

        # 5. Яркий оранжево-красный блик снизу слева.
        yy, xx = np.mgrid[0:height, 0:width]
        x_norm = xx / max(width - 1, 1)
        y_norm = yy / max(height - 1, 1)

        distance = np.sqrt(
            (x_norm / 0.42) ** 2
            + ((1 - y_norm) / 0.38) ** 2
        )
        leak = np.clip(1 - distance, 0, 1) ** 2

        # Усиленное локальное свечение в углу.
        extra_leak = 0.65 * np.exp(-(
            (x_norm / 0.28) ** 2
            + ((1 - y_norm) / 0.22) ** 2
        ))

        leak = np.maximum(leak, extra_leak).astype(np.float32)

        # Размываем маску, чтобы переходы оставались мягкими.
        leak = box_blur(leak)
        leak = box_blur(leak)
        leak = box_blur(leak)

        # BGR: яркий оранжево-красный свет.
        result[:, :, 0] += leak * 45
        result[:, :, 1] += leak * 110
        result[:, :, 2] += leak * 190

        return np.clip(result, 0, 255).astype(np.uint8)




def box_blur(image):
    """Размывает изображение усреднением соседних пикселей 3x3."""

    if image.ndim not in (2, 3):
        raise ValueError("Ожидается двумерное или цветное изображение")

    if image.ndim == 2:
        padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    else:
        padded = np.pad(
            image,
            ((1, 1), (1, 1), (0, 0)),
            mode="edge"
        )

    result = np.zeros_like(image, dtype=np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neighborhood = padded[y:y + 3, x:x + 3]
            result[y, x] = np.mean(neighborhood, axis=(0, 1))

    return result





class Neon(ImageFilter):
    """Создаёт неоновое свечение на основных контурах изображения."""

    def apply(self, image, strength=2.0, blur=1.3, **kwargs):
        if strength < 0:
            raise ValueError("strength не может быть отрицательным")
        if blur < 0:
            raise ValueError("blur не может быть отрицательным")

        original = image.astype(np.float32)

        # 1. Переводим изображение в яркость (порядок каналов BGR).
        gray = (
            0.114 * original[:, :, 0]
            + 0.587 * original[:, :, 1]
            + 0.299 * original[:, :, 2]
        )

        # 2. Немного сглаживаем яркость, чтобы уменьшить мелкие детали.
        smooth = box_blur(gray)
        smooth = box_blur(smooth)

        # 3. Вычисляем изменения яркости по горизонтали и вертикали.
        dx = np.zeros_like(smooth)
        dy = np.zeros_like(smooth)

        dx[:, 1:-1] = (smooth[:, 2:] - smooth[:, :-2]) / 2
        dy[1:-1, :] = (smooth[2:, :] - smooth[:-2, :]) / 2

        edges = np.sqrt(dx ** 2 + dy ** 2)

        # 4. Используем более высокий порог — мелких линий станет меньше.
        threshold = np.percentile(edges, 78)
        max_edge = np.percentile(edges, 99)

        if max_edge > threshold:
            edges = np.clip(
                (edges - threshold) / (max_edge - threshold),
                0,
                1,
            )
        else:
            edges = np.zeros_like(edges)

        # Немного усиливаем выбранные контуры.
        edges = edges ** 0.8

        # 5. Создаём фиолетовый слой свечения (BGR).
        glow = np.zeros_like(original)
        glow[:, :, 0] = edges * 210
        glow[:, :, 2] = edges * 255

        # 6. Размываем только слой свечения.
        blurred_glow = glow.copy()
        iterations = min(max(int(blur * 3), 0), 10)

        for _ in range(iterations):
            blurred_glow = box_blur(blurred_glow)

        # 7. Сохраняем исходную яркость и накладываем свечение.
        result = original.copy()
        result += blurred_glow * strength * 0.65

        # 8. Добавляем тонкую яркую линию по центру контура.
        core = np.zeros_like(original)
        core[:, :, 0] = edges * 220
        core[:, :, 1] = edges * 100
        core[:, :, 2] = edges * 255

        result += core * 0.8

        return np.clip(result, 0, 255).astype(np.uint8)
