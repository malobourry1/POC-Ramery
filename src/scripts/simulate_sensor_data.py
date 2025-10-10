"""Simule des données de capteur de température."""

import json
import math
import time

import requests

from src.constants import URL_TEMPERATURE_SENSOR

interval = 5


def oscillating_value(t: float) -> float:
    """Retourne une valeur oscillante entre 20 et 25 basée sur le temps t."""
    return 22.5 + 2.5 * math.sin(2 * math.pi * t / 60)


print("Envoi des données en continu... (Ctrl+C pour arrêter)")
t0 = time.time()
try:
    while True:
        t = time.time() - t0
        value = oscillating_value(t)
        payload = [{"temperature_value": round(value, 2)}]

        # Envoyer les données seulement si l'URL est configurée
        if URL_TEMPERATURE_SENSOR is not None:
            try:
                response = requests.post(
                    URL_TEMPERATURE_SENSOR,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=2,
                )
                if response.status_code != 200:
                    print(f"Erreur HTTP {response.status_code}: {response.text}")
            except requests.RequestException as e:
                print(f"Erreur d envoi : {e}")
        else:
            print(
                f"URL_TEMPERATURE_SENSOR non configurée. Données simulées : {payload}"
            )

        time.sleep(interval)

except KeyboardInterrupt:
    print("\nArrêt de l envoi des données.")
