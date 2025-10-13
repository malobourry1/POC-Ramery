"""Main constants of the project."""

import os
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).parents[1]

# Tandem
URL_CAR_SENSOR = os.environ.get("URL_CAR_SENSOR")
URL_TEMPERATURE_SENSOR = os.environ.get("URL_TEMPERATURE_SENSOR")
URL_WATER_FILLRATE_SENSOR = os.environ.get("URL_WATER_FILLRATE_SENSOR")
TANDEM_VEHICLE_COUNT_PARAMETER_NAME = "count_vehicle_value"

# YOLO
CONF_THRESH = 0.3
TARGET_CLASSES = ["mini-car"]
MIN_FRAMES_VISIBLE = 10
MIN_MISSING_FRAMES = 10


# GCP
PROJECT_ID = "ramery-poc-theodo"
TOPIC_ID = "sensor-data"
BIGQUERY_DATASET = "sensor_data"
BIGQUERY_TABLE = "raw-sensor-data"
SEND_INTERVAL = 3.0
