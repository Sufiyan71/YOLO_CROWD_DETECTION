
# detection_utils.py
import cv2
import torch
import numpy as np
import logging

# Get the 'detection.utils' logger. It will inherit handlers from the 'detection' logger.
logger = logging.getLogger('detection.utils')

try:
    from utils.general import non_max_suppression
except ImportError as e:
    logger.error(f"Failed to import YOLOv5 utilities: {e}", exc_info=True)
    raise

def preprocess_roi_image(roi_image, target_size):
    """Preprocess ROI image safely for model inference."""
    try:
        if roi_image is None or roi_image.size == 0: return None
        h, w = roi_image.shape[:2]
        if h == 0 or w == 0: return None
        
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(roi_image, (new_w, new_h))
        
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded, scale, (x_offset, y_offset)
    except Exception as e:
        logger.error(f"Error preprocessing ROI image: {e}", exc_info=True)
        return None

def detect_in_roi(model, roi_image, device, conf_thres, iou_thres, target_size):
    """Run detection on a single ROI, with robust error handling."""
    try:
        processed_result = preprocess_roi_image(roi_image, target_size)
        if processed_result is None: return []
        
        processed_img, scale, (x_offset, y_offset) = processed_result
        
                #img_tensor = torch.from_numpy(processed_img.transpose(2, 0, 1)).to(device).float() / 255.0
        img_tensor = torch.from_numpy(processed_img.transpose(2, 0, 1)).to(device) / 255.0
        img_tensor = img_tensor.type_as(next(model.parameters()))
        if img_tensor.ndimension() == 3: img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0])
        
        detections = []
        if pred is not None and len(pred) > 0:
            det = pred[0]
            if scale == 0: return []
            for *xyxy, conf, cls in det:
                if int(cls) == 0:
                    x1 = (xyxy[0] - x_offset) / scale
                    y1 = (xyxy[1] - y_offset) / scale
                    x2 = (xyxy[2] - x_offset) / scale
                    y2 = (xyxy[3] - y_offset) / scale
                    detections.append([x1, y1, x2, y2, float(conf)])
        return detections
    except Exception as e:
        logger.error(f"Error during detection in ROI: {e}", exc_info=True)
        return []

def extract_roi_region(image, polygon):
    """Extract ROI region safely from image using polygon mask."""
    try:
        if len(polygon) < 3: return None, None
        poly_array = np.array(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly_array)
        
        img_h, img_w = image.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_w - x), min(h, img_h - y)
        if w <= 0 or h <= 0: return None, None

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly_array], 255)
        
        roi_region = cv2.bitwise_and(image, image, mask=mask)
        return roi_region[y:y+h, x:x+w], (x, y)
    except Exception as e:
        logger.error(f"Error extracting ROI region: {e}", exc_info=True)
        return None, None