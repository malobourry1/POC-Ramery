"""Module utilitaire pour la gestion des capteurs."""

import json
import time
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
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


def display_frame_on_camera(
    frame: cv2.Mat | npt.NDArray[Any], x1: int, y1: int, x2: int, y2: int, name: str, conf: float
) -> None:
    """Displays a rectangle with a label on the camera feed."""
    label = f"{name} {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )


def display_information_on_camera(
    frame: cv2.Mat | npt.NDArray[Any], boxes: list[tuple[Any, ...]], fps_time: float
) -> None:
    """Displays information on the camera feed."""
    fps = 1.0 / (time.time() - fps_time) if time.time() - fps_time > 0 else 0.0
    fps_time = time.time()
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  Detected: {len(boxes)}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )


def extract_boxe_attribute(
    boxe: Any, model: Any, conf_threshold: float, target_classes: list[str]
) -> tuple[bool, int, int, int, int, str, float]:
    """Extract attributes from a bounding box."""
    conf = float(boxe.conf[0]) if hasattr(boxe.conf, "__len__") else float(boxe.conf)
    cls_id = int(boxe.cls[0]) if hasattr(boxe.cls, "__len__") else int(boxe.cls)
    name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
    if conf < conf_threshold:
        return False, 0, 0, 0, 0, "", 0.0
    if name in target_classes:
        xyxy = (
            boxe.xyxy[0].cpu().numpy()
            if hasattr(boxe.xyxy, "cpu")
            else np.array(boxe.xyxy[0])
        )
        x1, y1, x2, y2 = map(int, xyxy)
    else:
        return False, 0, 0, 0, 0, "", 0.0
    return True, x1, y1, x2, y2, name, conf
