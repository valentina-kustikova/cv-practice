import cv2 as cv
import sys
import argparse
from pathlib import Path
import numpy as np


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image_path',
                        help='Path to image.',
                        type=Path,
                        required=True,
                        dest='image_path')
    parser.add_argument('-m', '--model',
                        help='Path to yolox-tiny.onnx.',
                        type=Path,
                        required=True,
                        dest='model_path')

    args = parser.parse_args()

    return args


class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = self._get_image()

    def _get_image(self):
        if self.image_path.exists():
            return cv.cvtColor(cv.imread(self.image_path), cv.COLOR_BGR2RGB)
        else:
            raise ValueError('Incorrect psth to image.')

    def preproccess_image(self, mean, std):
        self.image = cv.dnn.blobFromImage(self.original_image, size=(608, 608))
        print(self.image.shape)
        self.image[0][0, :, :] -= mean[0]
        self.image[0][1, :, :] -= mean[1]
        self.image[0][2, :, :] -= mean[2]
        self.image[0][0, :, :] /= std[0]
        self.image[0][1, :, :] /= std[1]
        self.image[0][2, :, :] /= std[2]
        return self.image


class YoloxTiny:
    def __init__(self, model_path, image):
        self.model_path = model_path
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.net = self._get_net_from_path()
        self.image = image
        self.input = image.preproccess_image(self.mean, self.std)

    def _get_net_from_path(self):
        if self.model_path.exists():
            return cv.dnn.readNetFromONNX(self.model_path.absolute())
        else:
            raise ValueError('Incorrect path for model.')

    def set_input(self):
        self.net.setInput(self.input)

    def _sigmoid(self, x) -> float:
        return 1 / (1 + np.exp(-x))

    def inference(self):
        self.result = self.net.forward()
        print(self.result)

    def output_process(self):
        raw_out = self.result[0]
        print(raw_out.shape)
        print(self.image.original_image.shape)
        #scales = {'W': 416 / self.image.original_image.shape[0], 'H': 416 / self.image.original_image.shape[1]}
        for elem in raw_out:
            x, y = elem[0], elem[1]
            h, w = elem[2], elem[3]
            box_score = elem[4]
            predicted_class = np.max(elem[5::])
            num_of_class = np.argmax(elem[5::])
            if box_score >= 0.9:
                print(x, y, h, w, box_score, predicted_class, num_of_class)








def main():
    args = cli_argument_parser()
    img = Image(args.image_path)
    yolo = YoloxTiny(args.model_path, img)
    yolo.set_input()
    yolo.inference()
    yolo.output_process()


if __name__=='__main__':
    sys.exit(main() or 0)