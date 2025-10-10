"""detect_toy_car_yolo.py
Détection en temps réel via la caméra du Mac avec YOLOv8 (ultralytics).
Affiche les boîtes englobantes et la classe. Touche q pour quitter.
"""  # noqa: D205

import os

import cv2
from ultralytics import YOLO

from src.utils.sensors_utils import (
    display_frame_on_camera,
    display_information_on_camera,
    extract_boxe_attribute,
    send_value_to_url,
)

model = YOLO("models/fine-tunning-for-mini-cars.pt")
URL_CAR_SENSOR = os.environ.get("URL_CAR_SENSOR")

CONF_THRESH = 0.3
TARGET_CLASSES = ["mini-car"]
MIN_FRAMES_VISIBLE = 10
MIN_MISSING_FRAMES = 10
TANDEM_PARAMETER_NAME = "count_vehicle_value"


def main() -> None:
    """Main function."""
    active_cars = {}
    disappeared_cars = {}
    count = 0
    for results in model.track(
        source=0,
        conf=CONF_THRESH,
        tracker="bytetrack.yaml",
        stream=True,
        show=True,
        persist=True,
    ):
        frame = results.orig_img
        current_ids = set()
        boxes = []
        if hasattr(results, "boxes") and results.boxes is not None:
            for boxe in results.boxes:
                is_boxe_detected, x1, y1, x2, y2, name, conf, track_id = (
                    extract_boxe_attribute(boxe, model, CONF_THRESH, TARGET_CLASSES)
                )
                if not is_boxe_detected:
                    continue
                boxes.append((x1, y1, x2, y2, name, conf, track_id))

                if track_id != -1:
                    current_ids.add(track_id)
                    if track_id not in active_cars:
                        active_cars[track_id] = {"frames_visible": 1, "counted": False}
                    active_cars[track_id]["frames_visible"] += 1

        for x1, y1, x2, y2, name, conf, track_id in boxes:
            display_frame_on_camera(frame, x1, y1, x2, y2, name, conf)
            cv2.putText(
                frame,
                f"ID:{track_id}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
        # Détecter les voitures disparues
        disappeared_ids = [
            tid for tid in list(active_cars.keys()) if tid not in current_ids
        ]
        for tid in disappeared_ids:
            if tid not in disappeared_cars:
                disappeared_cars[tid] = {"frames_missing": 1, **active_cars[tid]}
            else:
                disappeared_cars[tid]["frames_missing"] += 1

            if disappeared_cars[tid]["frames_missing"] > MIN_MISSING_FRAMES:
                if (
                    not disappeared_cars[tid]["counted"]
                    and disappeared_cars[tid]["frames_visible"] >= MIN_FRAMES_VISIBLE
                ):
                    count += 1
                    disappeared_cars[tid]["counted"] = True
                    print(f"✅ Voiture comptée ! (ID {tid}) → Total = {count}")

                del active_cars[tid]
                del disappeared_cars[tid]

        display_information_on_camera(frame, count)

        cv2.imshow("Detection voiture miniature (YOLOv8)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        send_value_to_url(URL_CAR_SENSOR, TANDEM_PARAMETER_NAME, count)


if __name__ == "__main__":
    main()
