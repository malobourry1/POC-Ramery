"""Fonctions utiles pour les fonctions Google Cloud."""

import json
from datetime import datetime, timedelta
from typing import cast

import requests
from google.cloud import bigquery

from constants import (  # type: ignore
    CAR_DATA_BQ_TABLE_NAME,
    TEMPERATURE_DATA_BQ_TABLE_NAME,
    URL_CAR_SENSOR,
    URL_TEMPERATURE_SENSOR,
    URL_WATER_FILLRATE_SENSOR,
    WATER_FILLRATE_DATA_BQ_TABLE_NAME,
)


def send_value_to_url(url: str, parameter_name: str, parameter_value: float) -> None:
    """Envoie une valeur au format JSON à une URL donnée via une requête POST."""
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps({parameter_name: parameter_value}),
            timeout=2,
        )
        if response.status_code != 200:
            print(f"Erreur HTTP {response.status_code}: {response.text}")
    except requests.RequestException as e:
        print(f"Erreur d envoi : {e}")


def extract_and_prepare_data(
    payload: dict[str, object] | list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object], str | None, str]:
    """Extract and prepare data from payload for BigQuery insertion."""
    if isinstance(payload, list) and len(payload) > 0:
        data = cast("dict[str, object]", payload[0])
        print(f"Données extraites du tableau : {data}")
    else:
        data = cast("dict[str, object]", payload)
        print(f"Données directes : {data}")

    tandem_url: str | None = None
    row_to_insert_in_bq: dict[str, object] = {"time": None}
    bq_table: str = TEMPERATURE_DATA_BQ_TABLE_NAME

    if "temperature_value" in data:
        row_to_insert_in_bq["temperature_value"] = data["temperature_value"]
        tandem_url = URL_TEMPERATURE_SENSOR
        bq_table = TEMPERATURE_DATA_BQ_TABLE_NAME
    elif "count_vehicle_value" in data:
        row_to_insert_in_bq["count_vehicle_value"] = data["count_vehicle_value"]
        tandem_url = URL_CAR_SENSOR
        bq_table = CAR_DATA_BQ_TABLE_NAME
    elif "RainWaterFillPercentage_value" in data:
        row_to_insert_in_bq["RainWaterFillPercentage_value"] = data[
            "RainWaterFillPercentage_value"
        ]
        tandem_url = URL_WATER_FILLRATE_SENSOR
        bq_table = WATER_FILLRATE_DATA_BQ_TABLE_NAME

    row_to_insert_in_bq["time"] = (datetime.now() + timedelta(hours=2)).isoformat()

    return row_to_insert_in_bq, data, tandem_url, bq_table


def insert_data_in_bq_table(
    project_id: str, dataset_id: str, table_id: str, data_to_insert: dict[str, object]
) -> None:
    """Insert data in BigQuery table using gcloud CLI."""
    try:
        bq_client = bigquery.Client()
        table_id = f"{project_id}.{dataset_id}.{table_id}"
        errors = bq_client.insert_rows_json(table_id, [data_to_insert])
        if errors:
            print(f"Erreur BigQuery : {errors}")
        else:
            print("Insertion réussie dans BigQuery")
    except Exception as e:
        print(f"Erreur BigQuery : {e}")
