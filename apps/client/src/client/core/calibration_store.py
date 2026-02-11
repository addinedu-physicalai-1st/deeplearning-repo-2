"""시선 캘리브레이션 데이터 로컬 JSON 저장/불러오기 유틸리티."""

import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CALIBRATION_DIR = os.path.expanduser("~/.focusguard")


def _calibration_path(user_id: int) -> str:
    return os.path.join(CALIBRATION_DIR, f"gaze_calibration_{user_id}.json")


def save_gaze_calibration(user_id: int, calibration_data: list,
                          screen_width: int, screen_height: int) -> bool:
    """Save gaze calibration data to ~/.focusguard/gaze_calibration_{user_id}.json."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    payload = {
        "calibration_data": calibration_data,
        "screen_width": screen_width,
        "screen_height": screen_height,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = _calibration_path(user_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Gaze calibration saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save gaze calibration: {e}")
        return False


def load_gaze_calibration(user_id: int) -> dict | None:
    """Load saved gaze calibration data. Returns dict or None if not found/invalid."""
    path = _calibration_path(user_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (isinstance(data.get("calibration_data"), list)
                and isinstance(data.get("screen_width"), int)
                and isinstance(data.get("screen_height"), int)):
            return data
        logger.warning(f"Gaze calibration file at {path} has invalid format")
        return None
    except Exception as e:
        logger.error(f"Failed to load gaze calibration: {e}")
        return None
