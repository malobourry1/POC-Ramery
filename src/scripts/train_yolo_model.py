from ultralytics import YOLO  # noqa: D100

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/dataset.yaml",
    epochs=25,
    imgsz=1024,
    name="fine-tunning-for-mini-cars",
)

model.save("models/fine-tunning-for-mini-cars.pt")
