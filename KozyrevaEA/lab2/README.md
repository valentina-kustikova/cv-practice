
# Object Detection using SSD MobileNet

## Requirements

Make sure you have the following libraries installed:
- OpenCV
- NumPy

You can install the required libraries using pip:
```bash
pip install -r requirements.txt
```

## Model Files

You will need the following model files for the object detection:
- `mobilenet_iter_73000.caffemodel`: The trained weights for the SSD MobileNet model.
- `deploy.prototxt`: The configuration file for the model.

## Usage

### Command-Line Arguments

You can run the script from the command line with the following arguments:

- `-i`, `--input`: Path to the input image or video (required).
- `-c`, `--confidence`: Confidence threshold for object detection (default: 0.5).

### Example Commands

To process an image:
```bash
python main.py -i dataset\test\img.png  -c 0.5
```

To process a video:
```bash
python main.py -i dataset\test\video.mp4 -c 0.5
```

## How It Works
1. **ObjectDetector Class**: Loads the model and processes images or video frames to detect objects.
2. **ImageProcessor Class**: Handles loading and processing images, drawing bounding boxes around detected objects.
3. **VideoProcessor Class**: Processes video frames in real-time, applying the object detection model frame by frame.
4. **Main Function**: Parses command-line arguments and invokes the appropriate processing based on the input type (image or video).

## Output

The detected objects will be displayed with bounding boxes and labels. The counts of detected objects for each class will also be printed to the console.
Press `k` to stop processing video.
