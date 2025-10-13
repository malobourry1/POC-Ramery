"""Module pour stocker des données dans BigQuery."""

import base64
import json
import os
from datetime import datetime

from google.cloud import bigquery

# Configuration depuis les variables d'environnement
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "ramery-poc-theodo")
BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "sensor_data")
BIGQUERY_TABLE = os.environ.get("BIGQUERY_TABLE", "raw-sensor-data")
URL_CAR_SENSOR = os.environ.get("URL_CAR_SENSOR")
URL_TEMPERATURE_SENSOR = os.environ.get("URL_TEMPERATURE_SENSOR")
URL_WATER_FILLRATE_SENSOR = os.environ.get("URL_WATER_FILLRATE_SENSOR")


def main(event, context):
    """Fonction Cloud pour traiter les messages Pub/Sub et stocker dans BigQuery."""
    if "data" not in event:
        print("Pas de champ 'data' dans l'événement")
        return

    raw_data = base64.b64decode(event["data"]).decode("utf-8")
    print(f"Raw data reçue : {raw_data}")

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError:
        try:
            # Essayer de corriger le format JavaScript vers JSON valide
            # Ajouter des guillemets autour des clés
            import re
            corrected_data = re.sub(r'(\w+):', r'"\1":', raw_data)
            print(f"Data corrigée : {corrected_data}")
            payload = json.loads(corrected_data)
        except json.JSONDecodeError as e:
            print(f"Erreur JSON après correction : {e}")
            print(f"Raw data : {raw_data}")
            return

    row_to_insert_in_bq = {
        "time": datetime.now().isoformat(),
        "temperature_value": 0.0,
        "count_vehicle_value": 0,
        "RainWaterFillPercentage_value": 0.0,
    }

    # Si payload est une liste, prendre le premier élément
    if isinstance(payload, list) and len(payload) > 0:
        data = payload[0]
        print(f"Données extraites du tableau : {data}")
    else:
        data = payload
        print(f"Données directes : {data}")

    tandem_url = None
    if "temperature_value" in data:
        row_to_insert_in_bq["temperature_value"] = data["temperature_value"]
        tandem_url = URL_TEMPERATURE_SENSOR
    elif "count_vehicle_value" in data:
        row_to_insert_in_bq["count_vehicle_value"] = data["count_vehicle_value"]
        tandem_url = URL_CAR_SENSOR
    elif "RainWaterFillPercentage_value" in data:
        row_to_insert_in_bq["RainWaterFillPercentage_value"] = data[
            "RainWaterFillPercentage_value"
        ]
        tandem_url = URL_WATER_FILLRATE_SENSOR
    else:
        print(f"Donnée inconnue : {data}")
        return

    # Sending to BigQuery
    try:
        bq_client = bigquery.Client()
        table_id = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
        errors = bq_client.insert_rows_json(table_id, [row_to_insert_in_bq])
        if errors:
            print(f"Erreur BigQuery : {errors}")
        else:
            print("Insertion réussie dans BigQuery")
    except Exception as e:
        print(f"Erreur BigQuery : {e}")

    # Sending to Tandem (optionnel)
    if tandem_url:
        print(f"Sending to tandem at {tandem_url}, data {data}")
        # Implémentation d'envoi à Tandem si nécessaire
    else:
        print("URL Tandem non configurée")
