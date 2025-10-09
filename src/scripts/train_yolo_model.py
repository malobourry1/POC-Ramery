import os  # noqa: D100

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/dataset.yaml",
    epochs=15,
    imgsz=640,
    name="mini_cars_detector",
)

# Créer le répertoire models s'il n'existe pas
os.makedirs("models", exist_ok=True)
model.save("models/mini_cars_best.pt")
