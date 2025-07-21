# detects.py
# Main entry point for the standalone, interactive application.

import argparse
import logging
import time
import cv2
from pathlib import Path

# Import from our central utilities and config modules
from core_utils import apply_windows_path_patch, setup_directories, setup_logging
from config import config_standalone, load_partial_config_from_json
config = load_partial_config_from_json("config.json", config_standalone)

from roi_module import PolygonROIManager
from detection_module import detect

# --- Initial Setup ---
apply_windows_path_patch()
setup_directories()
setup_logging('standalone')

# --- Configuration ---
WINDOW_NAME = 'Crowd Density Analysis - Standalone'
ROI_CONFIG_FILE = "roi_config.json"

def main():
    parser = argparse.ArgumentParser()
    # Arguments will now override the defaults in the imported 'config_standalone' object
    parser.add_argument('--weights', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--img-size', type=int, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, help='IOU threshold for NMS')
    parser.add_argument('--device', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', help='save results to project/name')
    parser.add_argument('--name', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--hide-dots', action='store_true', help='hide detection dots')
    parser.add_argument('--skip-frames', type=int, help='process every Nth frame')
    
    # Parse arguments directly into our imported config object.
    # This neatly overrides defaults from config.py with command-line args.
    parser.parse_args(namespace=config)
    
    logging.info(f"Starting standalone detection with config: {config}")
    logging.info("Controls: [e] Edit, [s] Save, [c] Clear, [q] Quit")

    # --- Setup for Interactive Mode ---
    roi_manager = PolygonROIManager()
    roi_config_path = Path(ROI_CONFIG_FILE)
    roi_manager.load_config(filename=roi_config_path)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, roi_manager.mouse_callback)

    # frame_handler remains the same...
    def frame_handler(frame):
        """Displays frame and handles all keyboard inputs."""
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if roi_manager.input_mode:
            roi_manager.handle_key_input(key)
        else:
            if key == ord('q'): raise KeyboardInterrupt
            elif key == ord('e'):
                roi_manager.edit_mode = not roi_manager.edit_mode
                logging.info(f"ROI Edit Mode: {'ON' if roi_manager.edit_mode else 'OFF'}")
            elif key == ord('s') and roi_manager.edit_mode:
                roi_manager.save_config(filename=roi_config_path)
                roi_manager.edit_mode = False
                logging.info(f"ROI config saved to '{roi_config_path}'. Edit mode OFF.")
            elif key == ord('c') and roi_manager.edit_mode:
                roi_manager.polygons.clear()
                roi_manager.roi_max_thresholds.clear()
                roi_manager.current_polygon.clear()
                logging.info("All ROIs cleared.")

    # --- Main Detection Loop ---
    is_live = config.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or config.source.isnumeric()
    while True:
        try:
            # Pass the final config object (defaults + CLI overrides) to detect
            detect(
                config=config,
                roi_manager_instance=roi_manager,
                output_frame_buffer=frame_handler,
                roi_config_path=roi_config_path
            )
            if not is_live:
                logging.info("Finished processing static source. Press 'q' to exit.")
                while (cv2.waitKey(100) & 0xFF) != ord('q'): pass
                raise KeyboardInterrupt
            break
        except KeyboardInterrupt:
            logging.info("Exiting program.")
            break
        except Exception as e:
            logging.getLogger('detection').error(f"Detection process crashed: {e}", exc_info=True)
            if is_live:
                logging.info("Reconnecting in 10 seconds...")
                time.sleep(10)
            else:
                break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()