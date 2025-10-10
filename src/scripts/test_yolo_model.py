import cv2  # noqa: D100
from ultralytics import YOLO

model = YOLO("models/fine-tunning-for-mini-cars.pt")
img_name = "PXL_20251009_124551905.jpg"

img = cv2.imread(f"dataset/images/val/{img_name}")
h, w, _ = img.shape

# Recadrage central (exemple)
x1, x2 = int(w * 0.1), int(w * 0.75)
y1, y2 = int(h * 0.25), int(h * 0.75)
roi = img[y1:y2, x1:x2]

results = model.predict(roi, imgsz=1024, conf=0.1, show=True, save=True, save_txt=True)
