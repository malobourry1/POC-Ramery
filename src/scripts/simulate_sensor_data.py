import time
import math
import requests
import json

# URL de destination
url = "https://:5BLXGdo4TSiXZfaB33y1QQ@eu.tandem.autodesk.com/api/v1/timeseries/models/urn:adsk.dtm:RcE1IDqYSLGLpmtoVha9Rg/streams/AQAAACJBLUpcr0ywnmQzrZqKLmwAAAAA"

# Envoi toutes les 100 ms (~10 fois par seconde)
interval = 10  

# Fonction pour générer une valeur oscillante entre 20 et 25
def oscillating_value(t):
    # Sinusoïde entre 20 et 25
    return 22.5 + 2.5 * math.sin(2 * math.pi * t / 5)

print("Envoi des données en continu... (Ctrl+C pour arrêter)")

t0 = time.time()
try:
    while True:
        t = time.time() - t0
        value = oscillating_value(t)
        payload = [{"temperature_value": round(value, 2)}]

        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=2
            )
            if response.status_code != 200:
                print(f"Erreur HTTP {response.status_code}: {response.text}")
        except requests.RequestException as e:
            print(f"Erreur d’envoi : {e}")

        time.sleep(interval)

except KeyboardInterrupt:
    print("\nArrêt de l’envoi des données.")
