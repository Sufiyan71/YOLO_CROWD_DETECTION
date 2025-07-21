import cv2
import torch
from models.experimental import attempt_load
from utils.general import check_img_size


def load_model(weights, device, imgsz):
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    model.to(device).eval()
    if device.type != 'cpu':
        model.half()
    return model, stride, imgsz



def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    return [x, y, w, h]

def scale_coords_safe(img1_shape, coords, img0_shape):
    """
    Scales coordinates (xyxy) from img1_shape to img0_shape
    """
    # Calculate gain and pad
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords

def draw_boxes(img, tracks, colors):
    from utils_2 import xywh_to_xyxy
    for t in tracks:
        box = xywh_to_xyxy(t[:4])
        track_id = int(t[4])
        label = f"ID {track_id}"
        color = colors[track_id % len(colors)]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def show_counts(img, active, total):
    cv2.putText(img, f'Active: {active}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.putText(img, f'Total: {total}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

