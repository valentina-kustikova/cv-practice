import os
import logging
from PIL import Image, ImageDraw


logging.basicConfig(
    filename="image_filters.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class Filters:
    def __init__(self, 
                 image_path: str) -> None:
        try:
            self.image = Image.open(image_path)
            self.width, self.height = self.image.size
            self.pixels = list(self.image.getdata())
            self.new_pixels = []
            logging.info(f"Image {image_path} successfully loaded. Size: {self.width}x{self.height}")
        except FileNotFoundError:
            logging.error(f"File not found: {image_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            raise
    

    def grayscale(self) -> Image.Image:
        """Применяет градации серого."""
        try:
            self.new_pixels = [
                (avg := int((pixel[0] + pixel[1] + pixel[2]) / 3), avg, avg)
                for pixel in self.pixels
            ]
            grayscale_img = Image.new("RGB", (self.width, self.height))
            grayscale_img.putdata(self.new_pixels)
            logging.info("Grayscale filter applied successfully.")
            return grayscale_img
        except Exception as e:
            logging.error(f"Error applying grayscale filter: {e}")
            raise


    def sepia(self) -> Image.Image:
        """Применяет эффект сепии."""
        try:
            self.new_pixels = [
                (
                    min(int((pixel[0] * 0.393) + (pixel[1] * 0.769) + (pixel[2] * 0.189)), 255),
                    min(int((pixel[0] * 0.349) + (pixel[1] * 0.686) + (pixel[2] * 0.168)), 255),
                    min(int((pixel[0] * 0.272) + (pixel[1] * 0.534) + (pixel[2] * 0.131)), 255),
                )
                for pixel in self.pixels
            ]
            sepia_img = Image.new("RGB", (self.width, self.height))
            sepia_img.putdata(self.new_pixels)
            logging.info("Sepia filter applied successfully.")
            return sepia_img
        except Exception as e:
            logging.error(f"Error applying sepia filter: {e}")
            raise


    def resize(self, new_width: int, new_height: int) -> Image.Image:
        """Изменяет разрешение изображения."""
        try:
            resized_img = Image.new("RGB", (new_width, new_height))
            for y in range(new_height):
                for x in range(new_width):
                    src_x = int(x * self.width / new_width)
                    src_y = int(y * self.height / new_height)
                    resized_img.putpixel((x, y), self.image.getpixel((src_x, src_y)))
            logging.info(f"Image resized to {new_width}x{new_height} successfully.")
            return resized_img
        except Exception as e:
            logging.error(f"Error resizing image: {e}")
            raise


    def vignette(self) -> Image.Image:
        """Применяет виньетку."""
        try:
            vignette_img = self.image.copy()
            draw = ImageDraw.Draw(vignette_img)
            for x in range(self.width):
                for y in range(self.height):
                    dx = x - self.width / 2
                    dy = y - self.height / 2
                    distance = (dx**2 + dy**2) ** 0.5
                    max_distance = (self.width**2 + self.height**2) ** 0.5 / 2
                    alpha = max(0, int(255 * (1 - distance / max_distance)))
                    pixel = self.image.getpixel((x, y))
                    vignette_pixel = tuple(int(channel * alpha / 255) for channel in pixel)
                    vignette_img.putpixel((x, y), vignette_pixel)
            logging.info("Vignette filter applied successfully.")
            return vignette_img
        except Exception as e:
            logging.error(f"Error applying vignette filter: {e}")
            raise


    def pixelate_area(self, top_left: tuple[int, int], 
                      bottom_right: tuple[int, int], 
                      pixel_size: int) -> Image.Image:
        """Пикселизует заданную область изображения."""
        try:
            pixelated_img = self.image.copy()
            for y in range(top_left[1], bottom_right[1], pixel_size):
                for x in range(top_left[0], bottom_right[0], pixel_size):
                    avg_color = [0, 0, 0]
                    count = 0
                    for yy in range(y, min(y + pixel_size, bottom_right[1])):
                        for xx in range(x, min(x + pixel_size, bottom_right[0])):
                            pixel = self.image.getpixel((xx, yy))
                            avg_color = [avg_color[i] + pixel[i] for i in range(3)]
                            count += 1
                    avg_color = tuple(c // count for c in avg_color)
                    for yy in range(y, min(y + pixel_size, bottom_right[1])):
                        for xx in range(x, min(x + pixel_size, bottom_right[0])):
                            pixelated_img.putpixel((xx, yy), avg_color)
            logging.info(f"Pixelation applied to area {top_left} - {bottom_right} with size {pixel_size}.")
            return pixelated_img
        except Exception as e:
            logging.error(f"Error applying pixelation: {e}")
            raise


if __name__ == "__main__":
    try:
        os.makedirs("results", exist_ok=True)

        imgs_path = [
            "imgs/test1.jpg",
            "imgs/test2.jpg", 
        ]

        methods = [
            ("grayscale", lambda f: f.grayscale(), "results/grayscale_{0}.jpg"),
            ("sepia", lambda f: f.sepia(), "results/sepia_{0}.jpg"),
            ("resize", lambda f: f.resize(200, 150), "results/resized_{0}.jpg"),
            ("vignette", lambda f: f.vignette(), "results/vignette_{0}.jpg"),
            ("pixelate", lambda f: f.pixelate_area((50, 50), (200, 200), 10), "results/pixelated_{0}.jpg"),
        ]

        for img_path in imgs_path:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            filters = Filters(img_path)
            
            for method_name, method_func, output_path in methods:
                result_img = method_func(filters)
                result_img.save(output_path.format(img_name))
        logging.info("All filters applied successfully.")
    except Exception as e:
        logging.critical(f"Critical error in processing: {e}")
