"""Main constants of the project."""

import os
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).parents[1]
URL_CAR_SENSOR = os.environ.get("URL_CAR_SENSOR")
URL_TEMPERATURE_SENSOR = os.environ.get("URL_TEMPERATURE_SENSOR")

CONF_THRESH = 0.3
TARGET_CLASSES = ["mini-car"]
MIN_FRAMES_VISIBLE = 10
MIN_MISSING_FRAMES = 10
TANDEM_VEHICLE_COUNT_PARAMETER_NAME = "count_vehicle_value"
