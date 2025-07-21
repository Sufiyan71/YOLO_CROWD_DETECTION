# detection_module.py
import time
import logging
import threading
import subprocess
import shutil
import sys
from pathlib import Path
import io
import requests
import cv2
import numpy as np
import torch
import os

# (boto3 and email imports can be removed from here if they aren't used directly)

logger = logging.getLogger('detection')
email_logger = logging.getLogger('email_sender')

from roi_module import PolygonROIManager
from detection_utils import detect_in_roi, extract_roi_region
from redis_utils import update_crowd_count, push_alert, insert_camera_data
from send_email import send_email_with_image 

try:
    from models.experimental import attempt_load
    from utils.datasets import LoadImages
    from utils.general import check_img_size, non_max_suppression, increment_path
    from utils.torch_utils import select_device, time_sync
except ImportError as e:
    logger.error(f"Failed to import YOLOv5 utilities: {e}", exc_info=True)
    raise

# The API_ENDPOINT is now sourced from the config object passed into detect()

def upload_screenshot_and_save_local_async(frame: np.ndarray, filename: str, api_endpoint: str) -> Path:
    """
    Encodes a frame, saves it locally, and uploads it to the API endpoint in a background thread.
    Returns the local path where the screenshot was saved.
    """
    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    local_image_path = screenshot_dir / filename
    
    try:
        cv2.imwrite(str(local_image_path), frame)
        logger.info(f"Screenshot saved locally at: {local_image_path}")

        def _upload_target():
            try:
                logger.info(f"Preparing to upload screenshot: {filename} to {api_endpoint}")
                is_success, buffer = cv2.imencode(".jpg", frame)
                if not is_success:
                    logger.error("Failed to encode frame to JPEG for upload.")
                    return
                byte_io = io.BytesIO(buffer)
                files = {'file': (filename, byte_io, 'image/jpeg')}
                response = requests.post(api_endpoint, files=files, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded screenshot {filename} to {api_endpoint}.")
                else:
                    logger.error(f"Failed to upload screenshot. Server responded with "
                                 f"status {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error while uploading screenshot: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"An unexpected error occurred during screenshot upload: {e}", exc_info=True)
        
        threading.Thread(target=_upload_target, daemon=True).start()

    except Exception as e:
        logger.error(f"Error saving screenshot locally or starting upload thread: {e}", exc_info=True)
    
    return local_image_path

# FFmpegStreamer class remains unchanged...
# detection_module.py
# (all other code like imports and the detect function can remain the same)
# ...

class FFmpegStreamer:
    """
    Manages a robust FFmpeg process for reading RTSP streams, with
    automatic reconnection and a dynamic 'reconnecting' overlay.
    """
    def __init__(self, source_url: str, width: int, height: int):
        self.source_url = source_url
        self.width = width
        self.height = height
        self.ffmpeg_process = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_frame_time = 0
        self.is_reconnecting = False
        # The 'reconnecting_frame' is now generated dynamically, so no need to pre-create it.

    def _build_ffmpeg_command(self):
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe: raise FileNotFoundError("FFmpeg executable not found.")
        return [
            ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
            "-i", self.source_url, "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", "30", "-"
        ]

    # --- NEW: Dynamic Reconnecting Overlay Generator ---
    def _create_reconnecting_overlay(self, frame: np.ndarray) -> np.ndarray:
        """

        Draws a dynamic 'reconnecting' animation on the given frame.
        If the input frame is None, it creates a new black frame.
        """
        # If there's no last frame, create a black one
        if frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Create a semi-transparent black overlay for text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        # Blend the overlay with the frame
        final_frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # --- 1. Animated Spinner ---
        center_x, center_y = self.width // 2, self.height // 2 - 60
        num_dots = 12
        current_time = time.time()
        
        for i in range(num_dots):
            angle = 2 * np.pi * i / num_dots - (current_time * 3) # Rotate over time
            
            # Position of the dot
            x = int(center_x + 40 * np.cos(angle))
            y = int(center_y + 40 * np.sin(angle))
            
            # Animate size and brightness
            # Brightness cycles with a sine wave based on position
            brightness = int(128 + 127 * np.sin(angle + (current_time * 3)))
            color = (brightness, brightness, brightness)
            
            # Size also cycles
            radius = int(3 + 3 * np.sin(angle + (current_time * 3)))
            if radius > 1: # Only draw if radius is positive
                cv2.circle(final_frame, (x, y), radius, color, -1)

        # --- 2. Animated Text ---
        num_periods = int(current_time * 2) % 4  # Cycles 0, 1, 2, 3
        text = "Reconnecting" + "." * num_periods
        
        font, font_scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 1.2, 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height // 2) + 60
        cv2.putText(final_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return final_frame

    def _reader_thread(self):
        bytes_per_frame = self.width * self.height * 3
        while self.running:
            logger.info(f"Starting FFmpeg process for {self.source_url}")
            
            # Set reconnecting flag immediately
            with self.lock:
                self.is_reconnecting = True

            self.ffmpeg_process = subprocess.Popen(
                self._build_ffmpeg_command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            
            # Update last frame time only when we start a new process
            self.last_frame_time = time.time()
            
            while self.running and self.ffmpeg_process.poll() is None:
                # Check for stream stall
                if time.time() - self.last_frame_time > 10:
                    logger.warning("Stream stalled (>10s without a frame). Terminating FFmpeg to reconnect.")
                    break # Exit inner loop to trigger reconnect logic

                # Read a chunk of data
                chunk = self.ffmpeg_process.stdout.read(bytes_per_frame)
                
                if len(chunk) == bytes_per_frame:
                    with self.lock:
                        # We have a valid frame, so we are no longer "reconnecting"
                        self.is_reconnecting = False
                        self.latest_frame = np.frombuffer(chunk, np.uint8).reshape((self.height, self.width, 3))
                        self.last_frame_time = time.time() # Reset stall timer
                else:
                    # If we don't get a full frame, wait a bit
                    time.sleep(0.01)

            logger.warning("FFmpeg process ended. Cleaning up and will retry.")
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            
            # Set reconnecting flag again before sleeping
            with self.lock:
                 self.is_reconnecting = True
                 
            if self.running:
                logger.info("Waiting 5 seconds before reconnecting...")
                time.sleep(5)

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        if self.thread:
            self.thread.join()

    def read(self):
        with self.lock:
            # Check the reconnecting flag
            is_currently_reconnecting = self.is_reconnecting
            # Get a copy of the latest frame if it exists
            frame_to_use = self.latest_frame.copy() if self.latest_frame is not None else None

        if is_currently_reconnecting:
            # Generate the dynamic overlay. It will be drawn on the last good frame,
            # or on a black background if no frame has ever been received.
            reconnecting_frame = self._create_reconnecting_overlay(frame_to_use)
            return False, reconnecting_frame # Return False to indicate the frame is not "live"
        else:
            # We are not reconnecting, and we have a valid frame.
            return True, frame_to_use

# detection_module.py
# (Imports and other functions remain the same)
# ...

def detect(config, roi_manager_instance, output_frame_buffer, roi_config_path):
    try:
        source = config.source
        webcam = source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        device = select_device(config.device)
        model = attempt_load(config.weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(config.img_size, s=stride)
        if device.type != 'cpu': model.half()
        
        dataset = None
        if webcam:
            stream_width, stream_height = 1280, 720
            dataset = FFmpegStreamer(source, stream_width, stream_height)
            dataset.start()
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        last_global_alert_time, ALERT_COOLDOWN_SECONDS = 0, 120
        last_metadata_update_time, METADATA_UPDATE_INTERVAL = 0, 1

        processed_frames = 0
        logger.info("Starting main detection loop...")
        
        while True:
            im0 = None
            if webcam:
                ret, im0 = dataset.read()
                if im0 is None: time.sleep(0.1); continue
            else:
                try: _, _, im0s, _ = next(iter(dataset)); im0 = im0s.copy()
                except StopIteration: logger.info("Finished all files."); break
            
            if webcam and not ret:
                output_frame_buffer(im0)
                time.sleep(0.1)
                continue

            try:
                processed_frames += 1
                if processed_frames % config.skip_frames != 0: continue
                
                t_start = time_sync()
                overlay_dots = im0.copy()
                roi_counts, total_detections, statuses = [], 0, []
                
                for roi_idx, polygon in enumerate(roi_manager_instance.polygons):
                    roi_region, roi_offset = extract_roi_region(im0, polygon)
                    count = 0
                    if roi_region is not None:
                        detections = detect_in_roi(model, roi_region, device, config.conf_thres, config.iou_thres, imgsz)
                        for det in detections:
                             raw_center_x, raw_center_y = (det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0
                             center_x, center_y = raw_center_x + roi_offset[0], raw_center_y + roi_offset[1]
                             if roi_manager_instance.point_in_polygon((int(center_x), int(center_y)), polygon):
                                count += 1
                                if not config.hide_dots: cv2.circle(overlay_dots, (int(center_x), int(center_y)), 6, (0, 255, 0), -1)
                    
                    roi_counts.append(count)
                    total_detections += count
                    is_overflow = count >= roi_manager_instance.roi_max_thresholds[roi_idx] if roi_idx < len(roi_manager_instance.roi_max_thresholds) else False
                    statuses.append('overflow' if is_overflow else 'ok')

                # --- START OF MODIFICATION AREA ---

                # First, prepare the final display frame with all overlays.
                # This ensures the screenshot will have all the UI elements.
                if not config.hide_dots:
                    im0_with_dots = cv2.addWeighted(overlay_dots, 0.5, im0, 0.5, 0)
                else:
                    im0_with_dots = im0
                
                im0_with_rois = roi_manager_instance.draw_overlays(im0_with_dots, roi_counts)
                im0_final_display = roi_manager_instance.draw_count_display(im0_with_rois, roi_counts, total_detections)

                # Now, check for alert conditions
                current_time = time.time()
                total_max_threshold = sum(roi_manager_instance.roi_max_thresholds)
                
                if (total_detections > 0 and total_max_threshold > 0 and
                    total_detections >= total_max_threshold and
                    (current_time - last_global_alert_time > ALERT_COOLDOWN_SECONDS)):
                    
                    last_global_alert_time = current_time
                    logger.warning(f"SYSTEM-WIDE OVERFLOW: Total count ({total_detections}) exceeds total threshold ({total_max_threshold}). Alerting.")
                    ts = time.strftime('%Y_%m_%d_%H_%M_%S')
                    filename = f"alert_cam_{config.cam_id}_{ts}.jpg"
                    
                    # MODIFIED LINE: Pass the final processed frame for the screenshot
                    screenshot_local_path = upload_screenshot_and_save_local_async(im0_final_display.copy(), filename, config.API_ENDPOINT)
                    
                    base_url = config.API_ENDPOINT.replace('/upload/', '')
                    screenshot_url = f"{base_url}/uploads/{filename}"
                    
                    alert_msg = f"High crowd density detected at {config.location_name}"
                    
                    push_alert(
                        cam_id=config.cam_id, name=config.location_name, count=total_detections,
                        message=alert_msg, screenshot_path=screenshot_url, max_threshold_total=total_max_threshold
                    )

                    email_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    send_email_with_image(
                        location=config.location_name,
                        threshold=total_max_threshold,
                        timestamp=email_timestamp,
                        image_path=str(screenshot_local_path),
                        count=total_detections,
                        sender_email=config.email_sender_address,
                        recipient_emails=config.email_recipients
                    )
                
                # Update continuous crowd count
                update_crowd_count(config.cam_id, total_detections)
                
                if current_time - last_metadata_update_time > METADATA_UPDATE_INTERVAL:
                    if not insert_camera_data(config.cam_id, total_detections):
                        logger.warning(f"Failed to update camera metadata heartbeat for '{config.cam_id}'.")
                    last_metadata_update_time = current_time

                processing_time = time_sync() - t_start
                log_msg = f"Frame {processed_frames} ({processing_time:.3f}s) | Total: {total_detections} (Threshold: {total_max_threshold}) | ROI counts: {roi_counts} | status: {statuses}"
                logger.info(log_msg)

                # Finally, send the fully processed frame to the output buffer for display
                output_frame_buffer(im0_final_display)

                # --- END OF MODIFICATION AREA ---

            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
            if webcam:
                time.sleep(0.01)
                
    except Exception as e:
        logger.error(f"Critical error in detection core: {e}", exc_info=True)
        if 'dataset' in locals() and webcam:
            dataset.stop()
        raise e