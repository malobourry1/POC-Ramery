"""detect_toy_car_yolo.py
Détection en temps réel via la caméra du Mac avec YOLOv8 (ultralytics).
Affiche les boîtes englobantes et la classe. Touche 'q' pour quitter.
"""  # noqa: D205

import time

import cv2
import numpy as np
from ultralytics import YOLO

# Charger un modèle YOLOv8 léger (téléchargé automatiquement la première fois)
model = YOLO("yolov8n.pt")  # 'n' = nano, rapide

# Seuil de confiance et classes d'intérêt
CONF_THRESH = 0.30
TARGET_CLASSES = [
    "car",
    "truck",
    "bus",
]  # on cible 'car' — peut inclure d'autres classes


def main():
    """Main function."""
    cap = cv2.VideoCapture(0)  # 0 = caméra principale
    if not cap.isOpened():
        print(
            "Impossible d'ouvrir la caméra. Vérifie les permissions macOS (Préférences Système → Sécurité)."
        )
        return

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO attend généralement BGR images — on peut lui donner directement frame
        results = model(frame, imgsz=640, conf=CONF_THRESH, verbose=False)[0]

        boxes = []
        if hasattr(results, "boxes") and results.boxes is not None:
            for b in results.boxes:
                conf = float(b.conf[0]) if hasattr(b.conf, "__len__") else float(b.conf)
                cls_id = int(b.cls[0]) if hasattr(b.cls, "__len__") else int(b.cls)
                name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                if conf < CONF_THRESH:
                    continue
                # Si la classe fait partie des cibles
                if name in TARGET_CLASSES:
                    xyxy = (
                        b.xyxy[0].cpu().numpy()
                        if hasattr(b.xyxy, "cpu")
                        else np.array(b.xyxy[0])
                    )
                    x1, y1, x2, y2 = map(int, xyxy)
                    boxes.append((x1, y1, x2, y2, name, conf))

        # Dessiner boîtes
        for x1, y1, x2, y2, name, conf in boxes:
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

        # Afficher FPS et nombre de détections
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

        cv2.imshow("Détection voiture miniature (YOLOv8)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
