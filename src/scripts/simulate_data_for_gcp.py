"""Script pour publier des messages sur Pub/Sub via gcloud."""

import json
import subprocess
import sys


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


if __name__ == "__main__":
    project_id = "ramery-poc-theodo"
    topic_id = "sensor-topic"

    # Données à publier
    data = {"temperature_value": 23.93}

    print(f"Envoi du message : {data}")
    print(f"Vers le topic : {topic_id} dans le projet : {project_id}")

    success = publish_to_pubsub(project_id, topic_id, data)
    sys.exit(0 if success else 1)
