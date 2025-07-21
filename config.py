# config.py
# Central configuration file for the Crowd Density Analysis application.

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    """Single source of truth for all application parameters."""
    
    # --- Core Detection Parameters ---
    weights: List[str] = field(default_factory=lambda: ['best.pt'])
    source: str = 'data/images'
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = ''
    classes: List[int] = field(default_factory=lambda: [0])
    agnostic_nms: bool = False
    augment: bool = False
    
    # --- Application Behavior ---
    cam_id: int = 1
    location_name: str = 'Default Location'
    skip_frames: int = 1
    
    # --- Output & Saving ---
    project: str = 'runs/detect'
    name: str = 'exp'
    exist_ok: bool = False
    nosave: bool = False
    hide_dots: bool = False

    # --- Redis Configuration ---
    REDIS_HOST: str = '10.0.3.26'
    REDIS_PORT: int = 6379

    # --- API Endpoint ---
    API_ENDPOINT: str = "http://192.168.100.191:8000/upload/"

    # --- Email Alerting Configuration ---
    email_sender_address: str = "youremail@work.in"
    email_recipients: List[str] = field(default_factory=list)
    
    # --- AWS SES Credentials ---
    # WARNING: For production, use IAM roles or environment variables instead of hardcoding.
    AWS_REGION: str = 'ap-south-1'
    AWS_ACCESS_KEY_ID: Optional[str] = 'YOUR_AWS_ACCESS_KEY'  # <-- IMPORTANT: REPLACE OR USE ENV VAR
    AWS_SECRET_ACCESS_KEY: Optional[str] = 'YOUR_AWS_SECRET_ACCESS_KEY' # <-- IMPORTANT: REPLACE OR USE ENV VAR


# --- Pre-configured instance for the FastAPI Web Application ---
config_web = Config(
    source='rtsp://admin:Admin123@172.16.0.62:554/ch1/stream1',
    location_name='Mahadwar Chowk',
    cam_id=1,
    email_recipients=['shardul7x1@gmail.com', 'sachin.m@moodscope.in', 'deshmukhb92@gmail.com', 'rahulmehere.23@gmail.com'],
    img_size=416,
    conf_thres=0.25,
    iou_thres=0.5,
    device="cpu",
    nosave=True,
    skip_frames=2,
    project='runs/detect_web'
)

# --- Pre-configured instance for the Standalone (detects.py) Application ---
# This serves as a default and can be overridden by command-line arguments.
config_standalone = Config(
    source='0', # Default to webcam
    location_name='Standalone Test',
    cam_id=99,
    device='cpu',
    project='runs/detect_standalone',
    # Email recipients can be added here if needed for standalone testing
    # email_recipients=['test@example.com'],
)

import json

def load_partial_config_from_json(json_path: str, base_config: Config) -> Config:
    """Update only provided fields in config.json into a base Config object."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
    except Exception as e:
        print(f"Error loading config from JSON: {e}")
    return base_config
