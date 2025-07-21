# app.py
# Main FastAPI application for Crowd Density Analysis.
 
import uvicorn
import cv2
import logging
import threading
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
 
# Import from our central utilities and config modules
from core_utils import apply_windows_path_patch, setup_directories, setup_logging
from config import config_web, load_partial_config_from_json
config = load_partial_config_from_json("config.json", config_web)
 
 
# --- Initial Setup ---
apply_windows_path_patch()
setup_directories()
setup_logging('web_app')
 
# Import our custom modules AFTER logging is set up
from roi_module import PolygonROIManager
from detection_module import detect
 
# --- Application Configuration ---
# THE CONFIGURATION BLOCK IS NOW REMOVED. WE USE THE IMPORTED `config` OBJECT.
 
# --- Global Variables & Locks ---
output_frame = None
lock = threading.Lock()
roi_manager = PolygonROIManager()
central_roi_config_path = Path("roi_config.json")
 
# --- FastAPI App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
 
logging.info("=== CROWD DENSITY ANALYZER WEB APP STARTED ===")
logging.info(f"Using configuration: {config}")
 
def run_detection_thread():
    """Runs the YOLOv5 detection in a separate, resilient thread."""
    global output_frame, roi_manager, lock, central_roi_config_path
   
    detection_logger = logging.getLogger('detection')
    detection_logger.info("Detection thread starting...")
   
    while True:
        try:
            detection_logger.info(f"Attempting to start detection on source: {config.source}")
            # Pass the imported config object directly to the detect function
            detect(
                config=config,
                roi_manager_instance=roi_manager,
                output_frame_buffer=lambda frame: set_output_frame(frame),
                roi_config_path=central_roi_config_path
            )
        except KeyboardInterrupt:
            detection_logger.info("Detection thread stopped by user.")
            break
        except Exception as e:
            detection_logger.error(f"Detection thread crashed: {e}", exc_info=True)
            is_live = config.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
            if is_live:
                detection_logger.info("Live stream connection lost. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                detection_logger.error("Error processing static source. Stopping thread.")
                break
 
# The rest of app.py remains unchanged...
def set_output_frame(frame):
    """Safely update the global output frame."""
    global output_frame, lock
    with lock:
        output_frame = frame.copy()
 
def generate_video_stream():
    """Generator function that yields JPEG-encoded frames for the web stream."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            flag, encodedImage = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')
        time.sleep(1 / 30) # Stream at ~30 FPS
 
@app.on_event("startup")
def startup_event():
    logging.info("=== APPLICATION STARTUP EVENT ===")
    try:
        roi_manager.load_config(filename=central_roi_config_path)
    except Exception as e:
        logging.error(f"Could not load ROI config on startup: {e}")
    thread = threading.Thread(target=run_detection_thread, daemon=True)
    thread.start()
    logging.info("Detection thread started successfully.")
 
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logging.info(f"Serving main page to client: {request.client.host}")
    return templates.TemplateResponse("index.html", {"request": request})
 
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video_stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
 
@app.post("/toggle_edit_mode")
async def toggle_edit_mode():
    roi_manager.edit_mode = not roi_manager.edit_mode
    status = "ON" if roi_manager.edit_mode else "OFF"
    logging.info(f"Toggled ROI edit mode: {status}")
    return {"status": "success", "edit_mode": roi_manager.edit_mode}
 
@app.post("/save_rois")
async def save_rois():
    roi_manager.save_config(filename=central_roi_config_path)
    roi_manager.edit_mode = False
    logging.info("ROI configuration saved successfully.")
    return {"status": "success", "message": "ROIs saved."}
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_config=None)