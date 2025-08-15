import os
import cv2
import yaml

# ==== Load YOLO classes from data.yaml ====
with open("roadmap1.v3i.yolov8/data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)

all_classes = data_yaml["names"]

# Find indices of classes ending with '_rank'
rank_class_ids = [i for i, name in enumerate(all_classes) if name.endswith("_rank")]
print("Rank class IDs:", rank_class_ids)
print("Rank class names:", [all_classes[i] for i in rank_class_ids])

# ==== Paths ====
yolo_images_path = "roadmap1.v3i.yolov8/train/images"
yolo_labels_path = "roadmap1.v3i.yolov8/train/labels"
output_path = "rank_crops_unlabeled"
os.makedirs(output_path, exist_ok=True)

# ==== Loop through labels ====
for label_file in os.listdir(yolo_labels_path):
    if not label_file.endswith(".txt"):
        continue
    
    with open(os.path.join(yolo_labels_path, label_file), "r") as f:
        lines = f.readlines()
    
    img_file = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(yolo_images_path, img_file)
    if not os.path.exists(img_path):
        img_file = img_file.replace(".jpg", ".png")
        img_path = os.path.join(yolo_images_path, img_file)
    if not os.path.exists(img_path):
        continue
    
    image = cv2.imread(img_path)
    if image is None:
        continue
    h, w, _ = image.shape
    
    for idx, line in enumerate(lines):
        cls, x_center, y_center, width, height = map(float, line.strip().split())
        cls = int(cls)

        # Skip non-rank detections
        if cls not in rank_class_ids:
            continue
        
        class_name = all_classes[cls]
        
        # Convert YOLO coords to pixel coords
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        save_name = f"{label_file.replace('.txt','')}_{class_name}_{idx}.jpg"
        save_path = os.path.join(output_path, save_name)
        cv2.imwrite(save_path, crop)

print("✅ Crops saved in:", output_path)
