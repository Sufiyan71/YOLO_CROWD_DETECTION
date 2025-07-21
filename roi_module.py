# roi_module.py
# Contains classes for managing ROIs without any GUI library dependencies.

import cv2
import numpy as np
import json
import logging
import time
import os
from datetime import datetime

logger = logging.getLogger('detection.roi')

class PolygonROIManager:
    """Manages ROI creation and interaction directly on the OpenCV frame."""
    def __init__(self, logo_path="logo.jpeg"):
        # ROI data
        self.polygons = []
        self.roi_max_thresholds = []
        self.current_polygon = []
        
        # Interaction state
        self.edit_mode = False
        self.drawing = False
        self.selected_polygon = -1
        self.drag_point_index = -1
        self.mouse_pos = (0, 0)

        # --- NEW: State for non-blocking input ---
        self.input_mode = False
        self.input_buffer = ""
        self.input_prompt = ""
        self.pending_polygon = None
        
        # Logo handling
        self.logo_path = logo_path
        self.logo_img = None
        self.logo_size = (250, 100)
        self.load_logo()

    def mouse_callback(self, event, x, y, flags, param):
        """Handles mouse events for drawing and editing ROIs."""
        self.mouse_pos = (x, y)
        if not self.edit_mode or self.input_mode:
            return  # Disable mouse actions while getting user input

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                # Check if a point on an existing polygon is clicked for dragging
                for poly_idx, polygon in enumerate(self.polygons):
                    for point_idx, point in enumerate(polygon):
                        if np.sqrt((x - point[0])**2 + (y - point[1])**2) < 10:
                            self.selected_polygon, self.drag_point_index = poly_idx, point_idx
                            return
                # Start drawing a new polygon
                self.current_polygon = [(x, y)]
                self.drawing = True
            else:
                self.current_polygon.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN and self.drawing and len(self.current_polygon) >= 3:
            # --- MODIFIED: Enter input mode instead of showing a dialog ---
            self.pending_polygon = self.current_polygon.copy()
            self.current_polygon, self.drawing = [], False
            self.input_mode = True
            self.input_prompt = f"Enter Threshold for ROI #{len(self.polygons) + 1} (then press Enter):"
            self.input_buffer = ""

        elif event == cv2.EVENT_MOUSEMOVE and self.selected_polygon != -1:
            self.polygons[self.selected_polygon][self.drag_point_index] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_polygon, self.drag_point_index = -1, -1

    def handle_key_input(self, key):
        """
        Processes keyboard input when in input_mode.
        This is called from the main loop in `detects.py`.
        """
        if not self.input_mode:
            return

        # Enter key (ASCII 13)
        if key == 13:
            try:
                # Use default value if buffer is empty
                threshold = int(self.input_buffer) if self.input_buffer else 50
            except ValueError:
                threshold = 50 # Default if input is not a valid number
            
            self.polygons.append(self.pending_polygon)
            self.roi_max_thresholds.append(threshold)
            logger.info(f"ROI {len(self.polygons)} created with max threshold: {threshold}")
            self.cancel_input()

        # Backspace key (ASCII 8)
        elif key == 8:
            self.input_buffer = self.input_buffer[:-1]
        
        # Escape key (ASCII 27)
        elif key == 27:
            logger.info("ROI creation cancelled.")
            self.cancel_input()

        # Alphanumeric keys
        elif 32 <= key < 127:
            char = chr(key)
            if char.isdigit():
                self.input_buffer += char

    def cancel_input(self):
        """Resets the input mode state."""
        self.input_mode = False
        self.input_buffer = ""
        self.input_prompt = ""
        self.pending_polygon = None

    def draw_input_prompt(self, img):
        """Draws the input prompt on the screen if in input_mode."""
        if not self.input_mode:
            return

        h, w = img.shape[:2]
        # Create a semi-transparent black box for the prompt
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Display the prompt and the current input buffer
        prompt_text = f"{self.input_prompt} {self.input_buffer}"
        cv2.putText(img, prompt_text, (20, h - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_overlays(self, img, roi_counts):
        # ... (This function remains mostly the same) ...
        img_with_overlay = img.copy()
        # --- ADDED: Draw the input prompt if active ---
        self.draw_input_prompt(img_with_overlay)

        # The rest of the drawing logic...
        blink_overlay = img.copy()
        is_blinking = False
        blink_alpha = 0.8 if (time.time() * 2) % 2 > 1 else 0.4

        for i, polygon in enumerate(self.polygons):
            if len(polygon) >= 3 and i < len(roi_counts) and i < len(self.roi_max_thresholds):
                if roi_counts[i] >= self.roi_max_thresholds[i]:
                    is_blinking = True
                    cv2.fillPoly(blink_overlay, [np.array(polygon, dtype=np.int32)], (0, 0, 255))
        
        if is_blinking:
            img_with_overlay = cv2.addWeighted(img, 1 - blink_alpha, blink_overlay, blink_alpha, 0)
        
        if self.edit_mode:
            if self.current_polygon:
                for i, point in enumerate(self.current_polygon):
                    cv2.circle(img_with_overlay, point, 8, (0, 255, 0), -1)
                    if i > 0: cv2.line(img_with_overlay, self.current_polygon[i-1], point, (0, 255, 0), 3)
                if self.drawing: cv2.line(img_with_overlay, self.current_polygon[-1], self.mouse_pos, (0, 255, 0), 2)
            for polygon in self.polygons:
                cv2.polylines(img_with_overlay, [np.array(polygon, dtype=np.int32)], True, (255, 255, 0), 2)
                for point in polygon: cv2.circle(img_with_overlay, point, 10, (255, 0, 0), -1)
        
        self.draw_logo(img_with_overlay)
        return img_with_overlay

    # The rest of the PolygonROIManager class (load_logo, draw_logo, draw_count_display, save/load_config, point_in_polygon)
    # remains unchanged from the previous version. I'm omitting it here for brevity but it should be kept.

    def load_logo(self):
        try:
            if os.path.exists(self.logo_path):
                logo_original = cv2.imread(self.logo_path, cv2.IMREAD_UNCHANGED)
                if logo_original is not None:
                    self.logo_img = cv2.resize(logo_original, self.logo_size)
                    logger.info(f"Logo loaded from {self.logo_path}")
            else:
                logger.info(f"Logo file not found at {self.logo_path}.")
        except Exception as e:
            logger.error(f"Error loading logo: {e}")

    def draw_logo(self, img):
        try:
            if self.logo_img is None: return
            img_h, img_w = img.shape[:2]
            logo_h, logo_w = self.logo_img.shape[:2]
            margin = 20
            x_pos, y_pos = img_w - logo_w - margin, img_h - logo_h - margin
            if x_pos < 0 or y_pos < 0: return

            if self.logo_img.shape[2] == 4:
                logo_rgb = self.logo_img[:, :, :3]
                alpha = self.logo_img[:, :, 3] / 255.0
                for c in range(3):
                    img[y_pos:y_pos+logo_h, x_pos:x_pos+logo_w, c] = alpha * logo_rgb[:, :, c] + (1 - alpha) * img[y_pos:y_pos+logo_h, x_pos:x_pos+logo_w, c]
            else:
                overlay = img.copy()
                overlay[y_pos:y_pos+logo_h, x_pos:x_pos+logo_w] = self.logo_img
                cv2.addWeighted(overlay[y_pos:y_pos+logo_h, x_pos:x_pos+logo_w], 0.8, img[y_pos:y_pos+logo_h, x_pos:x_pos+logo_w], 0.2, 0, img[y_pos:y_pos+logo_h, x_pos:x_pos+logo_w])
        except Exception as e:
            logger.error(f"Error drawing logo: {e}")

    def point_in_polygon(self, point, polygon):
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (int(point[0]), int(point[1])), False) >= 0

    # roi_module.py

# ... (other functions in the class are fine) ...

    def draw_count_display(self, img, roi_counts, total_count):
        """Draws the time and count display panel and returns the modified image."""
        try:
            # If the input image is None for any reason, return it immediately to prevent a crash.
            if img is None:
                logger.warning("draw_count_display received a None image. Skipping.")
                return None

            panel_x, panel_y, panel_w, panel_h = img.shape[1] - 420, 20, 400, 120
            overlay = img.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 3)
            
            time_str = datetime.now().strftime("%H:%M:%S")
            cv2.putText(img, f"TIME: {time_str}", (panel_x + 20, panel_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(img, f"TOTAL COUNT: {total_count}", (panel_x + 20, panel_y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            
            
            return img

        except Exception as e:
            logger.error(f"Error drawing count display: {e}", exc_info=True)
           
            return img

    def save_config(self, filename="roi_config.json"):
        try:
            with open(filename, 'w') as f:
                json.dump({"polygons": self.polygons, "max_thresholds": self.roi_max_thresholds}, f, indent=2)
            logger.info(f"ROI configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save ROI config: {e}")

    def load_config(self, filename="roi_config.json"):
        try:
            with open(filename, 'r') as f: config = json.load(f)
            self.polygons = config.get("polygons", [])
            self.roi_max_thresholds = config.get("max_thresholds", [])
            
            if len(self.roi_max_thresholds) != len(self.polygons):
                self.roi_max_thresholds = ([50] * len(self.polygons))
                logger.warning("Mismatch in loaded ROI config. Resetting all thresholds to 50.")
            logger.info(f"ROI config loaded from {filename}. Found {len(self.polygons)} polygons.")
        except FileNotFoundError:
            logger.warning(f"No config file found: {filename}. Starting with no ROIs.")
        except Exception as e:
            logger.error(f"Error loading ROI config file '{filename}': {e}")