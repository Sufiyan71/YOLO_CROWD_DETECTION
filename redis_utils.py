# redis_utils.py
# Encapsulates all Redis-related operations with robust logging and error handling.

import redis
import json
import time
from datetime import datetime
import logging
from typing import Optional

# Import the specific web config, as Redis is a shared service
# usually configured once for the whole system.
from config import config_web as config

logger = logging.getLogger('redis')

redis_client = None

def get_redis_client():
    """Establishes and returns a resilient Redis client connection."""
    global redis_client
    if redis_client:
        try:
            if redis_client.ping():
                return redis_client
        except redis.exceptions.ConnectionError:
            logger.warning("Redis connection lost. Attempting to reconnect...")
            redis_client = None
    try:
        # Use settings from the central config file
        client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=0, decode_responses=True)
        client.ping()
        logger.info(f"Successfully connected to Redis server at {config.REDIS_HOST}:{config.REDIS_PORT}.")
        redis_client = client
        return redis_client
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}: {e}")
        return None

# --- Redis Constants ---
MAX_ENTRIES = 6000

# --- Core Functions (no change in logic, just ensuring get_redis_client uses the new config) ---

def push_alert(
    cam_id: int,
    name: str,
    count: int,
    message: str,
    level: str = "critical",
    screenshot_path: Optional[str] = None,
    max_threshold_total: Optional[int] = None
) -> bool:
    """Pushes a rich alert to the 'alerts_queue' in Redis."""
    try:
        client = get_redis_client()
        if not client:
            return False

        timestamp = time.time()
        alert_id = client.incr("alert_counter")
        
        alert = {
            "id": alert_id,
            "cam_id": cam_id,
            "cam_name": name,
            "count": count,
            "message": message,
            "timestamp": timestamp,
            "level": level,
        }
        if screenshot_path:
            alert["screenshot_path"] = screenshot_path
        if max_threshold_total is not None:
            alert["threshold"] = max_threshold_total
            
        client.rpush("alerts_queue", json.dumps(alert))
        logger.warning(f"⚠️ Alert pushed: {alert}")
        return True
    except (redis.exceptions.RedisError, Exception) as e:
        logger.error(f"Failed to push alert for CAM_ID={cam_id}: {e}", exc_info=True)
        return False

def update_crowd_count(cam_id: int, count: int) -> bool:
    """Updates the rolling time series of crowd counts for a specific camera."""
    try:
        client = get_redis_client()
        if not client: return False
        
        timestamp = time.time()
        key = f"crowd_cam_id_{cam_id}"
        entry = {"timestamp": timestamp, "count": count}
        
        pipeline = client.pipeline()
        pipeline.lpush(key, json.dumps(entry))
        pipeline.ltrim(key, 0, MAX_ENTRIES - 1)
        pipeline.expire(key, MAX_ENTRIES * 2)
        pipeline.execute()
        
        logger.info(f"📊 Updated crowd count for CAM_'{cam_id}': {count}")
        return True
    except (redis.exceptions.RedisError, Exception) as e:
        logger.error(f"Failed to update crowd count for {cam_id}: {e}", exc_info=True)
        return False

def insert_camera_data(cam_id: int, count: int) -> bool:
    """
    Inserts or updates static metadata for a camera, acting as a heartbeat.
    This key stores the last known status of the camera.
    """
    try:
        client = get_redis_client()
        if not client:
            return False
            
        key = f"cam_id_{cam_id}"
        value = {
            "cam_id": cam_id,
            "count": count,
            "time": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        }
        client.set(key, json.dumps(value))
        logger.info(f"✅ Inserted metadata for CAM_ID={cam_id}")
        return True
    except (redis.exceptions.RedisError, Exception) as e:
        logger.error(f"Failed to insert metadata heartbeat for CAM_ID={cam_id}: {e}", exc_info=True)
        return False