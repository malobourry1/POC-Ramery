"""Module utilitaire pour la gestion des capteurs."""

import json
import subprocess

import requests


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


def publish_to_pubsub(project_id: str, topic_id: str, data: dict[str, float]) -> bool:
    """Publier un message via gcloud CLI."""
    try:
        message = json.dumps(data)
        cmd = [
            "gcloud",
            "pubsub",
            "topics",
            "publish",
            topic_id,
            f"--message={message}",
            f"--project={project_id}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Message publié avec succès !")
        print(f"Output: {result.stdout.strip()}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la publication : {e}")
        print(f"Stderr: {e.stderr}")
        return False
