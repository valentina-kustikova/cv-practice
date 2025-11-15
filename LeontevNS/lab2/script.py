import numpy as np
import cv2 as cv
import glob
import argparse
import asyncio
import threading

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from nanodet import NanoDet
from yolox import YoloX

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

class TPRCalculator:
    def __init__(self, threshold = 0.5):
        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.threshold = threshold
    def calculate_from_boxes(self, box1, box2):
        # Let classes be equal
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        x_l_rightest = max(x1, x3)
        y_l_rightest = max(y1, y3)
        x_r_leftest = min(x2, x4)
        y_r_leftest = min(y2, y4)
        intersection_area = max(0, x_r_leftest - x_l_rightest) * max(0, y_r_leftest - y_l_rightest)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def append_to_tp(self, value, class1, class2):
        if (value >= self.threshold and class1 == class2):
            self.true_positives += 1
            return True
        return False

    def get_tpr(self):
        if (self.true_positives != 0 or self.false_negatives != 0):
            return self.true_positives / (self.true_positives + self.false_negatives)
        return 0

    def get_fdr(self):
        if (self.true_positives != 0 or self.false_positives != 0):
            return self.false_positives / (self.true_positives + self.false_positives)
        return 0

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def get_tpr_on_frame(preds, frameid, tprcalc, letterbox_scale):
    flag = False
    flag2 = False
    with open('mov03478.txt', 'r') as file:
        content = file.readlines()
        for i in range(len(content)):
            data = content[i].split(sep=' ')
            data[-1] = data[-1][:-1]
            if (frameid == int(data[0])):
                flag2 = False
                flag = True
                className = data[1].lower()
                for pred in preds:
                    box1 = unletterbox(pred[:4], letterbox_scale).astype(np.int32)
                    x1 = box1[0]
                    x2 = box1[0] + box1[2]
                    y1 = box1[1]
                    y2 = box1[1] + box1[3]
                    box1 = np.array([x1, y1, x2, y2])
                    box2 = np.array([data[2], data[3], data[4], data[5]]).astype(np.int32)
                    if tprcalc.append_to_tp(tprcalc.calculate_from_boxes(box1, box2), classes[int(pred[5])], className):
                        flag2 = True
                if not flag2:
                    tprcalc.false_negatives += 1
            if (frameid != int(data[0]) and flag):
                break

def get_fdr_on_frame(preds, frameid, fdrcalc, letterbox_scale):
    flag = False
    flag2 = False
    lines = []
    with open('mov03478.txt', 'r') as file:
        content = file.readlines()
        for i in range(len(content)):
            data = content[i].split(sep=' ')
            data[-1] = data[-1][:-1]
            if (frameid == int(data[0])):
                lines.append(data)
                flag = True
            if (frameid != int(data[0]) and flag):
                break
    for pred in preds:
        flag2 = False
        box1 = unletterbox(pred[:4], letterbox_scale).astype(np.int32)
        x1 = box1[0]
        x2 = box1[0] + box1[2]
        y1 = box1[1]
        y2 = box1[1] + box1[3]
        box1 = np.array([x1, y1, x2, y2])
        for data in lines:
            className = data[1].lower()
            box2 = np.array([data[2], data[3], data[4], data[5]]).astype(np.int32)
            if fdrcalc.append_to_tp(fdrcalc.calculate_from_boxes(box1, box2), className, classes[int(pred[5])]):
                flag2 = True
        if not flag2:
            fdrcalc.false_positives += 1

def get_tpr_on_frame_nano(preds, frameid, tprcalc, imgshape, letterbox_scale):
    flag = False
    flag2 = False
    with open('mov03478.txt', 'r') as file:
        content = file.readlines()
        for i in range(len(content)):
            data = content[i].split(sep=' ')
            data[-1] = data[-1][:-1]
            if (frameid == int(data[0])):
                flag2 = False
                flag = True
                className = data[1].lower()
                for pred in preds:
                    box1 = unletterbox_nano(pred[:4], imgshape, letterbox_scale).astype(np.int32)
                    box2 = np.array([data[2], data[3], data[4], data[5]]).astype(np.int32)
                    if tprcalc.append_to_tp(tprcalc.calculate_from_boxes(box1, box2), classes[int(pred[5])], className):
                        flag2 = True
                if not flag2:
                    tprcalc.false_negatives += 1
            if (frameid != int(data[0]) and flag):
                break

def get_fdr_on_frame_nano(preds, frameid, fdrcalc, imgshape, letterbox_scale):
    flag = False
    flag2 = False
    lines = []
    with open('mov03478.txt', 'r') as file:
        content = file.readlines()
        for i in range(len(content)):
            data = content[i].split(sep=' ')
            data[-1] = data[-1][:-1]
            if (frameid == int(data[0])):
                lines.append(data)
                flag = True
            if (frameid != int(data[0]) and flag):
                break
    for pred in preds:
        flag2 = False
        box1 = unletterbox_nano(pred[:4], imgshape, letterbox_scale).astype(np.int32)
        for data in lines:
            className = data[1].lower()
            box2 = np.array([data[2], data[3], data[4], data[5]]).astype(np.int32)
            if fdrcalc.append_to_tp(fdrcalc.calculate_from_boxes(box1, box2), className, classes[int(pred[5])]):
                flag2 = True
        if not flag2:
            fdrcalc.false_positives += 1

async def letterbox_nano(srcimg, target_size=(416, 416)):
    img = srcimg.copy()

    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT, value=0)  # add border
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)

    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale

def unletterbox_nano(bbox, original_image_shape, letterbox_scale):
    ret = bbox.copy()

    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale

    if h == w:
        ratio = h / newh
        ret = ret * ratio
        return ret

    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)

    return ret.astype(np.int32)

async def vis_nano(preds, res_img, letterbox_scale):
    ret = res_img.copy()

    # draw bboxes and labels
    for pred in preds:
        bbox = pred[:4]
        conf = pred[-2]
        classid = pred[-1].astype(np.int32)
        if classid == 2:
            color = (255, 0, 0)
        elif classid == 5:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        # bbox
        xmin, ymin, xmax, ymax = unletterbox_nano(bbox, ret.shape[:2], letterbox_scale)
        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), color, thickness=2)

        # label
        label = "{:s}: {:.3f}".format(classes[classid], conf)
        cv.putText(ret, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)

    return ret
    
async def letterbox(srcimg, target_size=(640, 640)):
    padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
    ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
    resized_img = cv.resize(
        srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img

    return padded_img, ratio

def unletterbox(bbox, letterbox_scale):
    return bbox / letterbox_scale

async def vis(dets, srcimg, letterbox_scale):
    res_img = srcimg.copy()

    for det in dets:
        box = unletterbox(det[:4], letterbox_scale).astype(np.int32)
        score = det[-2]
        cls_id = int(det[-1])
        if cls_id == 2:
            color = (255, 0, 0)
        elif cls_id == 5:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        x0, y0, w, h = box

        text = '{}:{:.3f}%'.format(classes[cls_id], score * 100)
        font = cv.FONT_HERSHEY_SIMPLEX
        txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
        cv.rectangle(res_img, (x0, y0 , w, h), color, 2)
        cv.rectangle(res_img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), (0, 0, 0), -1)
        cv.putText(res_img, text, (x0, y0 + txt_size[1]), font, 0.4, (255, 255, 255), thickness=1)

    return res_img

def show_func(inp, img, time):
    cv.namedWindow(inp, cv.WINDOW_AUTOSIZE)
    cv.moveWindow(inp, 0, 0)
    for pic in img:
        cv.imshow(inp, pic)
        cv.waitKey(time)
    cv.destroyAllWindows()
    #cv.waitKey(0)

async def main():
    parser = argparse.ArgumentParser(description='Nanodet/Yolo inference')
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
    parser.add_argument('--input', '-i', type=str, default="imgs_MOV03478", help="Frames folder path")
    parser.add_argument('--confidence', default=0.5, type=float,
                        help='Class confidence')
    parser.add_argument('--nms', default=0.6, type=float,
                        help='Enter nms IOU threshold')
    parser.add_argument('--obj', default=0.5, type=float,
                        help='Enter object threshold')
    parser.add_argument('--nano', '-n', action='store_true', help='Should we use nano, false -> yolo')
    parser.add_argument('--start', '-s', default=0, type=int, help='Starting frame')
    parser.add_argument('--batch', '-b', default=50, type=int, help='Batch size')
    parser.add_argument('--threshold', '-t', default=0.5, type=float, help='Threshold for TP evaluation')
    args = parser.parse_args()

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    if args.nano:
    	model = NanoDet(modelPath= 'object_detection_nanodet_2022nov.onnx',
                    	prob_threshold=args.confidence,
                    	iou_threshold=args.nms,
                    	backend_id=backend_id,
                    	target_id=target_id)
    else:
        model = YoloX(modelPath= 'object_detection_yolox_2022nov.onnx',
                      confThreshold=args.confidence,
                      nmsThreshold=args.nms,
                      objThreshold=args.obj,
                      backendId=backend_id,
                      targetId=target_id)

    tm = cv.TickMeter()
    tm_est = cv.TickMeter()
    tm_est.reset()
    tm.reset()
    files = glob.glob(f"{args.input}/*.jpg")
    step = args.batch
    eps = 0.5
    timer_est = 0
    thread = None
    tpr_calc = TPRCalculator(args.threshold)
    tpr_calc_f = TPRCalculator(args.threshold)
    fdr_calc = TPRCalculator(args.threshold)
    fdr_calc_f = TPRCalculator(args.threshold)
    for count_batch in range(args.start, len(files), step):
        tm.reset()
        tm_est.reset()
        iter = (count_batch-args.start)//step + 1
        print(f"Batch #{iter} ({step} pictures)")
        tm_est.start()
        if (count_batch + step > len(files)):
            end = len(files)
        else:
            end = count_batch + step
        image = [cv.imread(file) for file in files[count_batch:end]]
        input_blob = [cv.cvtColor(file, cv.COLOR_BGR2RGB) for file in image]
        letterbox_scale = [None] * len(input_blob)
        # Letterbox transformation
        if args.nano:
            for index, blob in enumerate(input_blob):
                input_blob[index], letterbox_scale[index] = await letterbox_nano(blob)
        else:
            for index, blob in enumerate(input_blob):
                input_blob[index], letterbox_scale[index] = await letterbox(blob)

        # Inference
        preds = [None] * len(input_blob)
        tm.start()
        for index, file in enumerate(input_blob):
            preds[index] = await model.infer(file)
        tm.stop()
        print("Inference time: {:.2f} ms".format(tm.getTimeMilli()))
        img = [None] * len(image)
        if args.nano:
            for index, file in enumerate(image):
                get_tpr_on_frame_nano(preds[index], index+count_batch, tpr_calc, file.shape[:2], letterbox_scale[index])
                get_tpr_on_frame_nano(preds[index], index+count_batch, tpr_calc_f, file.shape[:2], letterbox_scale[index])
                get_fdr_on_frame_nano(preds[index], index+count_batch, fdr_calc, file.shape[:2], letterbox_scale[index])
                get_fdr_on_frame_nano(preds[index], index+count_batch, fdr_calc_f, file.shape[:2], letterbox_scale[index])
                img[index] = await vis_nano(preds[index], file, letterbox_scale[index])
        else:
            for index, file in enumerate(image):
                get_tpr_on_frame(preds[index], index+count_batch, tpr_calc, letterbox_scale[index])
                get_tpr_on_frame(preds[index], index+count_batch, tpr_calc_f, letterbox_scale[index])
                get_fdr_on_frame(preds[index], index+count_batch, fdr_calc, letterbox_scale[index])
                get_fdr_on_frame(preds[index], index+count_batch, fdr_calc_f, letterbox_scale[index])
                img[index] = await vis(preds[index], file, letterbox_scale[index])
        tm_est.stop()
        timer_est += tm_est.getTimeMilli()
        print(f'TPR For all frames: {tpr_calc.get_tpr()}')
        print(f'TPR For these {step} frames: {tpr_calc_f.get_tpr()}')
        print(f'FDR For all frames: {fdr_calc.get_fdr()}')
        print(f'FDR For these {step} frames: {fdr_calc_f.get_fdr()}')
        tpr_calc_f = TPRCalculator(args.threshold)
        fdr_calc_f = TPRCalculator(args.threshold)
        if thread != None:
            thread.join()
        thread = threading.Thread(target=show_func, args=(f"Image batch [{count_batch+1}-{end}]", img, int(((eps+1)*timer_est/(iter+1))//step),))
        thread.start()

if __name__ == '__main__':
    asyncio.run(main())
