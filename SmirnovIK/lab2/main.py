import YOLOv8Detector, MobileNetDetector, NanoDetDetector
from utils import read_annotations, class_color, load_class_names, evaluate_frame
import cv2, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--model", required=True, choices=["yolo","mob","nano"])
parser.add_argument("--conf_th", default=0.5)
parser.add_argument("--nms_th",  default=0.3)
parser.add_argument("--mode", required=True, choices=["metrics", "video"], default="video")
args=parser.parse_args()

gt = read_annotations("D:\\cv_labs\\2\\mov03478.txt")

if args.model=="yolo":
    det = YOLOv8Detector.YOLOv8Detector("yolov8n.onnx", args.conf_th, args.nms_th)
elif args.model=="mob":
    det = MobileNetDetector.MobileNetDetector("mob2\\frozen_inference_graph.pb", "mob2\\ssd_mobilenet_v2_coco_2018_03_29.pbtxt", args.conf_th, args.nms_th)
elif args.model=="nano":
    det = NanoDetDetector.NanoDetDetector("nanodet-plus-m_416.onnx", args.conf_th, args.nms_th)

detections={}
all_tpr, all_fdr, counter = 0,0,0
for i,f in enumerate(sorted(os.listdir(args.path))):
    img=cv2.imread(f"{args.path}/{f}")
    class_names = load_class_names("classes.txt")
    res = det.detect(img)
    gt_frame = gt.get(i, [])

    gt_boxes = [(cls, box) for cls, box in gt_frame]
    detections[i]=res
    TPR, FDR = evaluate_frame(
        [(gt_frame[i]["class"], gt_frame[i]["bbox"]) for i in range(len(gt_frame))],
        [(x,y,w,h,class_names[cid]) for x,y,w,h,conf,cid in res]
    )

    if args.mode =="video":
        for x,y,w,h,conf,cid in res:
            name = class_names[cid] if cid < len(class_names) else str(cid)
            cv2.rectangle(img,(x,y),(x+w,y+h),class_color(cid),2)
            cv2.putText(img,f"{name} {conf:.3f}",
                (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,class_color(cid),2)
        cv2.putText(img, f"TPR: {TPR:.3f}   FDR: {FDR:.3f}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
        cv2.imshow("det",img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("Завершение работы.")
            break
        elif key == ord(' '):
            print("Пауза. Нажмите любую клавишу чтобы продолжить.")
            cv2.waitKey(0)
    elif args.mode == "metrics":
        all_tpr += TPR
        all_fdr += FDR
        counter += 1

if args.mode == "metrics": print(f"TPR={all_tpr/counter:.3f}, FDR={all_fdr/counter:.3f}")
