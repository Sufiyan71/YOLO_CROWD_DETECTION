# core_utils.py
# Centralized utilities for configuration, logging, and system setup.

import logging
import platform
import sys
from pathlib import Path

# --- 1. Centralized System Patches and Directory Setup ---
def apply_windows_path_patch():
    if platform.system() == "Windows":
        import pathlib
        try:
            pathlib.PosixPath = pathlib.WindowsPath
        except AttributeError:
            pass

def setup_directories():
    dirs_to_create = ["logs", "screenshots", "runs/detect"]
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Required directories are ready: {dirs_to_create}")


# --- 2. Centralized Configuration Class ---
# THIS HAS BEEN MOVED TO config.py


# --- 3. Centralized Logging Setup (Unchanged) ---
def setup_logging(app_name: str):
    """
    Sets up comprehensive, separated logging for the entire application.
    - 'redis' logs -> 'logs/redis.log'
    - 'detection' logs -> 'logs/detection.log'
    - 'email_sender' logs -> 'logs/email_sender.log'
    - Main app logs -> 'logs/{app_name}.log'
    """
    log_dir = Path("logs")
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- Handlers ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    detection_log_file = log_dir / "detection.log"
    detection_file_handler = logging.FileHandler(detection_log_file, mode='a', encoding='utf-8')
    detection_file_handler.setFormatter(detailed_formatter)
    
    main_log_file = log_dir / f"{app_name}.log"
    main_file_handler = logging.FileHandler(main_log_file, mode='a', encoding='utf-8')
    main_file_handler.setFormatter(detailed_formatter)

    redis_log_file = log_dir / "redis.log"
    redis_file_handler = logging.FileHandler(redis_log_file, mode='a', encoding='utf-8')
    redis_file_handler.setFormatter(detailed_formatter)

    email_log_file = log_dir / "email_sender.log"
    email_file_handler = logging.FileHandler(email_log_file, mode='a', encoding='utf-8')
    email_file_handler.setFormatter(detailed_formatter)

    # --- Configure Loggers ---
    logging.getLogger().handlers = []
    
    redis_logger = logging.getLogger('redis')
    redis_logger.setLevel(logging.INFO)
    redis_logger.addHandler(redis_file_handler)
    redis_logger.addHandler(console_handler)
    redis_logger.propagate = False

    detection_logger = logging.getLogger('detection')
    detection_logger.setLevel(logging.INFO)
    detection_logger.addHandler(detection_file_handler)
    detection_logger.addHandler(console_handler)
    detection_logger.propagate = False

    email_logger_instance = logging.getLogger('email_sender')
    email_logger_instance.setLevel(logging.INFO)
    email_logger_instance.addHandler(email_file_handler)
    email_logger_instance.addHandler(console_handler)
    email_logger_instance.propagate = False

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(main_file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"=== LOGGING INITIALIZED FOR: {app_name.upper()} ===")
    logging.info(f"Main application logs -> {main_log_file.resolve()}")
    detection_logger.info(f"All detection logs -> {detection_log_file.resolve()}")
    redis_logger.info(f"All Redis logs -> {redis_log_file.resolve()}")
    email_logger_instance.info(f"All Email logs -> {email_log_file.resolve()}")