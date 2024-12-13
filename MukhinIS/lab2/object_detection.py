import cv2 as cv
import sys
import argparse
from pathlib import Path
import numpy as np
import copy

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image_path',
                        help='Path to image.',
                        type=Path,
                        required=False,
                        dest='image_path')
    parser.add_argument('-v', '--video_path',
                        help='Path to video.',
                        type=Path,
                        required=False,
                        dest='video_path')
    parser.add_argument('-m', '--model',
                        help='Path to yolox-tiny.onnx.',
                        type=Path,
                        required=True,
                        dest='model_path')
    parser.add_argument('-c', '--classes',
                        help='Path to classes file.',
                        type=Path,
                        required=True,
                        dest='classes_path')

    args = parser.parse_args()

    return args

class Video:
    def __init__(self, video_path, classes_name):
        self.video_path = video_path
        self.classes_name = classes_name
        self.frames = []
        if Path(self.classes_name).exists():
            with open(self.classes_name, 'r') as file:
                self.classes_name = file.readlines()
        else:
            raise ValueError('Incorrect path to the classes file.')

    def frame_detect(self, model):
        cap = cv.VideoCapture(self.video_path)
        while(cap.isOpened()): 
            ret, frame = cap.read()
            if frame is not None:
                model.image.add_frame(frame)
                model.set_input()
                model.inference()
                out = model.output_process()
                if out is not None:
                    final_boxes, final_scores, final_cls_inds = out
                    imm = model.image.vis(final_boxes, final_scores, final_cls_inds, self.classes_name)
                    self.frames.append(imm)
            else:
                break

    def gen_video(self):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video=cv.VideoWriter('video_detect.avi',fourcc, 20, (800,800))
        for elem in self.frames:
            video.write(elem)



class Image:
    def __init__(self, image_path):
        self.image_path = image_path
        if image_path is not None:
            self.original_image = self._get_image()

    def add_frame(self, frame):
        if frame.shape[0] > 800 or frame.shape[1] > 800:
            self.original_image = cv.resize(frame, (800, 800))
        else:
            self.original_image = frame

    def _get_image(self):
        if self.image_path.exists():
            tmp = cv.imread(self.image_path)
            if tmp.shape[0] > 800 or tmp.shape[1] > 800:
                return cv.resize(tmp, (800, 800))
        else:
            raise ValueError('Incorrect path to image.')

    def preproccess_image(self, mean, std):
        self.image = cv.dnn.blobFromImage(self.original_image, size=(416, 416))
        self.image[0][0, :, :] -= mean[0]
        self.image[0][1, :, :] -= mean[1]
        self.image[0][2, :, :] -= mean[2]
        self.image[0][0, :, :] /= std[0]
        self.image[0][1, :, :] /= std[1]
        self.image[0][2, :, :] /= std[2]
        return self.image

    def vis(self, boxes, scores, cls_ids, class_names=None, conf=0.5):
        new_image = np.array(self.original_image)
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv.FONT_HERSHEY_SIMPLEX

            txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
            cv.rectangle(new_image, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv.rectangle(
                new_image,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv.putText(new_image, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return new_image


class YoloxTiny:
    def __init__(self, model_path, image):
        self.model_path = model_path
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.net = self._get_net_from_path()
        self.image = image

    def _get_net_from_path(self):
        if self.model_path.exists():
            return cv.dnn.readNetFromONNX(self.model_path.absolute())
        else:
            raise ValueError('Incorrect path for model.')

    def set_input(self):
        self.input = self.image.preproccess_image(self.mean, self.std)
        self.net.setInput(self.input)

    def inference(self):
        self.result = self.net.forward()

    def nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        print(f'image size = {img_size[0]}')
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            print(hsize, wsize, stride)
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def output_process(self):
        raw_out = self.result
        predictions = self.demo_postprocess(raw_out[0], (416, 416))
        ratio = min(416 / self.image.original_image.shape[0], 416 / self.image.original_image.shape[1])
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = self.multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            return final_boxes, final_scores, final_cls_inds

def image_inference(image_path, model_path, classes_path):
    img = Image(image_path)
    yolo = YoloxTiny(model_path, img)
    if Path(classes_path).exists():
        with open(classes_path, 'r') as file:
            classes_name = file.readlines()
    else:
        raise ValueError('Incorrect path to the classes file.')
    classes_name = [line.rstrip('\n') for line in classes_name]
    yolo.set_input()
    yolo.inference()
    final_boxes, final_scores, final_cls_inds = yolo.output_process()
    imm = img.vis(final_boxes, final_scores, final_cls_inds, classes_name)
    while(True):
        cv.imshow('', imm)
        k = cv.waitKey(0)
        if k == ord('q'):
            break

def video_inference(video_path, model_path, classes_path):
    vid = Video(video_path, classes_path)
    img = Image(None)
    model = YoloxTiny(model_path, img)
    vid.frame_detect(model)
    vid.gen_video()


def main():
    args = cli_argument_parser()
    if str(args.image_path) is not None:
        image_inference(image_path=args.image_path, model_path=args.model_path, classes_path=args.classes_path)
    elif str(args.video_path) is not None:
        video_inference(args.video_path, args.model_path, args.classes_path)


if __name__=='__main__':
    sys.exit(main() or 0)