"""Module pour stocker des données dans BigQuery."""

import base64
import json
import os

from gcloud_utils import (
    extract_and_prepare_data,
    insert_data_in_bq_table,
    send_value_to_url,
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "ramery-poc-theodo")
BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "sensor_data")
BIGQUERY_TABLE = os.environ.get("BIGQUERY_TABLE", "raw-sensor-data")
URL_CAR_SENSOR = os.environ.get("URL_CAR_SENSOR")
URL_TEMPERATURE_SENSOR = os.environ.get("URL_TEMPERATURE_SENSOR")
URL_WATER_FILLRATE_SENSOR = os.environ.get("URL_WATER_FILLRATE_SENSOR")


def process_sensor_data(event: dict[str, object], context: object) -> None:
    """Fonction Cloud pour traiter les messages Pub/Sub et stocker dans BigQuery."""
    if "data" not in event:
        print("Pas de champ 'data' dans l'événement")
        return

    raw_data = base64.b64decode(str(event["data"])).decode("utf-8")
    print(f"Raw data reçue : {raw_data}")

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError:
        try:
            import re

            corrected_data = re.sub(r"(\w+):", r'"\1":', raw_data)
            print(f"Data corrigée : {corrected_data}")
            payload = json.loads(corrected_data)
        except json.JSONDecodeError as e:
            print(f"Erreur JSON après correction : {e}")
            print(f"Raw data : {raw_data}")
            return

    row_to_insert_in_bq, data_for_tandem, tandem_url = extract_and_prepare_data(
        payload=payload
    )

    insert_data_in_bq_table(
        project_id=PROJECT_ID,
        dataset_id=BIGQUERY_DATASET,
        table_id=BIGQUERY_TABLE,
        data_to_insert=row_to_insert_in_bq,
    )
    tandem_url = None
    if tandem_url:
        parameter_name, parameter_value = next(iter(data_for_tandem.items()))
        send_value_to_url(
            url=tandem_url,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
        )
    else:
        print("URL Tandem non configurée")
