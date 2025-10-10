"""detect_toy_car_yolo.py
Détection en temps réel via la caméra du Mac avec YOLOv8 (ultralytics).
Affiche les boîtes englobantes et la classe. Touche q pour quitter.
"""  # noqa: D205

import time

import cv2
from ultralytics import YOLO

from src.utils.sensors_utils import (
    display_frame_on_camera,
    display_information_on_camera,
    extract_boxe_attribute,
)

model = YOLO("models/mini_cars_clean_best.pt")

CONF_THRESH = 0.3
TARGET_CLASSES = ["mini-car"]

URL_CAR_SENSOR = ""


def main() -> None:
    """Main function."""
    cap = cv2.VideoCapture(0)
    print(model.names)
    if not cap.isOpened():
        print(
            "Impossible d ouvrir la caméra. Vérifie les permissions macOS (Préférences Système → Sécurité)."
        )
        return

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=CONF_THRESH, verbose=False)[0]

        boxes = []
        if hasattr(results, "boxes") and results.boxes is not None:
            for boxe in results.boxes:
                is_boxe_detected, x1, y1, x2, y2, name, conf = extract_boxe_attribute(
                    boxe, model, CONF_THRESH, TARGET_CLASSES
                )
                if not is_boxe_detected:
                    continue
                boxes.append((x1, y1, x2, y2, name, conf))

        for x1, y1, x2, y2, name, conf in boxes:
            display_frame_on_camera(frame, x1, y1, x2, y2, name, conf)

        display_information_on_camera(frame, boxes, fps_time)

        cv2.imshow("Détection voiture miniature (YOLOv8)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # send_value_to_url(URL_CAR_SENSOR, "count_cars", len(boxes))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
