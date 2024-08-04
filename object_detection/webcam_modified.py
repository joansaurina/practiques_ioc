import os
import cv2
import torch
from pathlib import Path
from ...yolov9.utils.general import (scale_boxes, check_img_size, non_max_suppression)
from ...yolov9.models.common import DetectMultiBackend
from ...yolov9.utils.torch_utils import select_device
from ...yolov9.utils.plots import Annotator, colors

# Cargar el modelo una vez
def load_model(weights, device):
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)  # check image size
    return model, imgsz, names

# Función para procesar cada frame
def process_frame(im0, model, imgsz, names, conf_thres=0.25, iou_thres=0.45):
    im = torch.from_numpy(im0).to(model.device)
    im = im.float() / 255.0  # uint8 to fp16/32
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=False)

    # NMS
    pred = non_max_suppression(pred[0], conf_thres, iou_thres, max_det=1000)

    detected_objects = set()
    annotator = Annotator(im0, line_width=3, example=str(names))
    
    for det in pred:  # detections per image
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                detected_objects.add(str(cls))
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

    im0 = annotator.result()
    return im0, detected_objects

def main():
    weights = 'best.pt'  # Define tu ruta de pesos aquí
    device = select_device('')  # 'cpu' or '0' for GPU
    model, imgsz, names = load_model(weights, device)

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    cap3 = cv2.VideoCapture(2)
    
    all_detected_objects = []

    while True:
        success1, img1 = cap1.read()
        success2, img2 = cap2.read()
        success3, img3 = cap3.read()

        if success1 and success2 and success3:
            img1, detected_objects1 = process_frame(img1, model, imgsz, names)
            img2, detected_objects2 = process_frame(img2, model, imgsz, names)
            img3, detected_objects3 = process_frame(img3, model, imgsz, names)

            all_detected_objects.append(detected_objects1)
            all_detected_objects.append(detected_objects2)
            all_detected_objects.append(detected_objects3)

            cv2.imshow('Webcam 1', img1)
            cv2.imshow('Webcam 2', img2)
            cv2.imshow('Webcam 3', img3)

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("Failed to capture frames from all cameras")
            break

    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()

    common_objects = set.intersection(*all_detected_objects)
    with open('label_summary.txt', 'w') as f:
        for obj in common_objects:
            f.write(f"{obj}\n")

if __name__ == '__main__':
    main()
