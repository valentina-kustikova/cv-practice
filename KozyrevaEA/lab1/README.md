# Image Filtering Script

## Requirements
- Python 3.10+
- OpenCV (`cv2`)
- NumPy

## Installation
1. Clone this repository or download the files.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the script with the following options:

```sh
python main.py --image_path <path_to_image> --filter <filter_name> [options]
```

### Available Filters and Options

#### Vignette Effect
```sh
python main.py --image_path imgs\test.jpg --filter vignette --radius 200 --intensity 1.0
```
- `--radius` (float): Radius of the vignette effect (default: 1.5)
- `--intensity` (float): Intensity of the effect (default: 1.0)

#### Pixelation
```sh
python main.py --image_path imgs\test.jpg --filter pixelate --pixel_size 10
```
- `--pixel_size` (int): Size of pixels for pixelation (default: 10)
- Requires selecting a region manually with the mouse.

#### Grayscale Conversion
```sh
python main.py --image_path imgs\test.jpg --filter grayscale
```

#### Resizing
```sh
python main.py --image_path imgs\test.jpg --filter resize --resize_width 200 --resize_height 300
```
- `--resize_width` (int): New width
- `--resize_height` (int): New height

#### Sepia Effect
```sh
python main.py --image_path imgs\test.jpg --filter sepia
```
