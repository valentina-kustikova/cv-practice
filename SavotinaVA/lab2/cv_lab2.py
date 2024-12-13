import sys
import argparse
import cv2 as cv
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", 
                        help="Path to source file (supported .jpg and .mp4)", 
                        required=True)
    parser.add_argument("-w", "--weight", 
                        help="Path to weights file", 
                        required=True)
    parser.add_argument("-cfg", "--config", 
                        help="Path to config file", 
                        required=True)
    parser.add_argument("-cl", "--classes", 
                        help="Path to classes file", 
                        required=True)
    parser.add_argument("-c", "--confidence", 
                        help="The confidence threshold (default = 0.5)", 
                        type=float, default=0.5)
    parser.add_argument("-n", "--nms", 
                        help="The threshold for suppressing non-maximums (default = 0.4)", 
                        type=float, default=0.4)
    parser.add_argument("-o", "--output", 
                        help="Write output to file")
    parser.add_argument("-v", "--view", 
                        help="View image/video in new window", 
                        action="store_true")
    return parser.parse_args()

class Detector:
    def __init__(self, args):
        self.args = args
        self.paths = self.make_paths(args)

    def make_paths(self, args):
        return {
            "src": args.source,
            "weight": args.weight,
            "config": args.config,
            "classes": args.classes,
            "output": args.output
        }
    
    def load_model(self):
        net = cv.dnn.readNet(self.paths["weight"], self.paths["config"])
        layer_names = net.getLayerNames()
        layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, layers

    def load_classes(self):
        with open(self.paths["classes"], "r") as f:
            return [line.strip() for line in f.readlines()]

    def detect_objects(self, image, net, classes, layers, colors):
        height, width, _ = image.shape
        blob = cv.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(layers)

        class_ids = []
        probabilities = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                probability = scores[class_id]
                if probability >= self.args.confidence:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    probabilities.append(float(probability))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, probabilities, self.args.confidence, self.args.nms)
        font = cv.FONT_HERSHEY_PLAIN
        objects_count = {}

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                confidence = probabilities[i]
                cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv.putText(image, f"{label} {confidence:.3f}", (x, y - 10), font, 1, color, 2)
                if label in objects_count:
                    objects_count[label] += 1
                else:
                    objects_count[label] = 1

        return image, objects_count

    def process_image(self, net, classes, layers, colors):
        image = cv.imread(self.paths["src"])
        output_img, objects_count = self.detect_objects(image, net, classes, layers, colors)
        
        print("Objects and objects counts:")
        if not objects_count:
            print("Objects are not found.")
        for obj, count in objects_count.items():
            print(f"{obj}: {count}")

        if self.paths["output"]:
            output_path = self.paths["output"]
            if not output_path.endswith(".jpg"):
                output_path += ".jpg"
            cv.imwrite(output_path, output_img)

        if self.args.view:
            cv.imshow("Image", output_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def process_video(self, net, classes, layers, colors):
        video_capture = cv.VideoCapture(self.paths["src"])
        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        fps = video_capture.get(cv.CAP_PROP_FPS)

        if self.paths["output"]:
            output_path = self.paths["output"]
            if not output_path.endswith(".mp4"):
                output_path += ".mp4"
            out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            output_frame, objects_count = self.detect_objects(frame, net, classes, layers, colors)
            print("Objects and objects counts:")
            if not objects_count:
                print("Objects are not found.")
            for obj, count in objects_count.items():
                print(f"{obj}: {count}")

            if self.paths["output"]:
                out.write(output_frame)

            if self.args.view:
                cv.imshow("Video", output_frame)
                if cv.waitKey(1) & 0xFF != 0xFF:
                    break

        video_capture.release()
        if self.paths["output"]:
            out.release()
        if self.args.view:
            cv.destroyAllWindows()

    def run(self):    
        net, layers = self.load_model()
        classes = self.load_classes()
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        if self.paths["src"].endswith(".jpg"):
            self.process_image(net, classes, layers, colors)
        elif self.paths["src"].endswith(".mp4"):
            self.process_video(net, classes, layers, colors)
        else:
            print("Unsupported file format. Please use .jpg or .mp4.")
            return 1

        return 0

def main():
    try:
        args = parse_arguments()
        app = Detector(args)
        return app.run()
    except Exception as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
    