import os  # noqa: D100

import cv2

IMAGES_DIR = "dataset/images/val"
LABELS_DIR = "dataset/labels/val"
SAVE_IMAGES_DIR = "dataset/images_cropped/val"
SAVE_LABELS_DIR = "dataset/labels_cropped/val"

os.makedirs(SAVE_IMAGES_DIR, exist_ok=True)
os.makedirs(SAVE_LABELS_DIR, exist_ok=True)

# Taille du crop (ex : zoom centre, 50% de l'image)
CROP_RATIO = 0.5

for img_name in os.listdir(IMAGES_DIR):
    if not img_name.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    H, W, _ = img.shape

    x1 = int(W * (1 - CROP_RATIO) / 2)
    x2 = int(W * (1 + CROP_RATIO) / 2)
    y1 = int(H * (1 - CROP_RATIO) / 2)
    y2 = int(H * (1 + CROP_RATIO) / 2)

    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(SAVE_IMAGES_DIR, img_name), crop_img)

    # Traitement du fichier label
    label_path = os.path.join(LABELS_DIR, os.path.splitext(img_name)[0] + ".txt")
    new_labels = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            lines = f.readlines()
        for line in lines:
            cls, x_c, y_c, w_box, h_box = map(float, line.strip().split())
            x_c_px = x_c * W
            y_c_px = y_c * H
            w_px = w_box * W
            h_px = h_box * H

            # Vérifier si la box est dans le crop
            x_min = x_c_px - w_px / 2
            x_max = x_c_px + w_px / 2
            y_min = y_c_px - h_px / 2
            y_max = y_c_px + h_px / 2

            if x_max < x1 or x_min > x2 or y_max < y1 or y_min > y2:
                continue  # box hors crop

            # Clip
            x_min_new = max(x_min, x1)
            x_max_new = min(x_max, x2)
            y_min_new = max(y_min, y1)
            y_max_new = min(y_max, y2)

            # Recalcul coordonnées normalisées
            new_W = x2 - x1
            new_H = y2 - y1
            x_c_new = (x_min_new + x_max_new) / 2 - x1
            y_c_new = (y_min_new + y_max_new) / 2 - y1
            w_new = x_max_new - x_min_new
            h_new = y_max_new - y_min_new

            x_c_new /= new_W
            y_c_new /= new_H
            w_new /= new_W
            h_new /= new_H

            new_labels.append(f"{int(cls)} {x_c_new} {y_c_new} {w_new} {h_new}\n")

    # Sauvegarde le nouveau fichier
    with open(
        os.path.join(SAVE_LABELS_DIR, os.path.splitext(img_name)[0] + ".txt"), "w"
    ) as f:
        f.writelines(new_labels)
