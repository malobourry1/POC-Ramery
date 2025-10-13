"""Simule des données de capteur de température."""

import sys
import time
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from constants import PROJECT_ID, SEND_INTERVAL, TOPIC_ID
from utils.info_sending_utils import publish_to_pubsub


def simulate_sensor_data(last_send_time: float) -> float:
    """Simule des données de capteur de température."""
    import random

    temperature_value = round(random.uniform(15.0, 30.0), 2)
    data = {"temperature_value": temperature_value}

    current_time = time.time()
    if current_time - last_send_time >= SEND_INTERVAL:
        try:
            publish_to_pubsub(
                project_id=PROJECT_ID,
                topic_id=TOPIC_ID,
                data=data,
            )
            print(f"📤 Température envoyée: {temperature_value}")
            return current_time
        except Exception as e:
            print(f"Erreur lors de l'envoi des données : {e}")
            return last_send_time
    return last_send_time


if __name__ == "__main__":
    last_send_time = time.time()
    while True:
        last_send_time = simulate_sensor_data(last_send_time=last_send_time)
        time.sleep(5)
