"""Constants for cloud functions."""

import os

URL_CAR_SENSOR = os.environ.get("URL_CAR_SENSOR")
URL_TEMPERATURE_SENSOR = os.environ.get("URL_TEMPERATURE_SENSOR")
URL_WATER_FILLRATE_SENSOR = os.environ.get("URL_WATER_FILLRATE_SENSOR")

TEMPERATURE_DATA_BQ_TABLE_NAME = "temperature-data"
CAR_DATA_BQ_TABLE_NAME = "count-car-data"
WATER_FILLRATE_DATA_BQ_TABLE_NAME = "level-rain-water-data"
