import os
import shutil
from ultralytics import YOLO

# === 1️⃣ Paths ===
# Path to your dataset (must follow YOLO classification folder format)
dataset_path = "rank_classifier"  # contains train/ and valid/ subfolders

# Base folder of your manual-yolo project
project_base = r"C:\Users\HP\manual-yolo"

# YOLO training output folder
project_runs = os.path.join(project_base, "runs")

# Final classifier model path
final_model_path = os.path.join(project_base, "rank_classifier.pt")

# === 2️⃣ Load YOLOv8 classification model (pretrained on ImageNet) ===
model = YOLO("yolov8n-cls.pt")  # "n" = nano, good for small datasets

# === 3️⃣ Train ===
model.train(
    data=dataset_path,
    epochs=100,
    imgsz=64,
    batch=64,
    workers=4,
    patience=10,
    augment=True,
    project=project_runs,         # Save inside manual-yolo\runs
    name="rank_classifier"        # Run name
)

# === 4️⃣ Auto-copy best.pt to manual-yolo/rank_classifier.pt ===
best_model_path = os.path.join(project_runs, "classify", "rank_classifier", "weights", "best.pt")

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, final_model_path)
    print(f"\n✅ Training complete! Best model copied to: {final_model_path}")
else:
    print("\n❌ ERROR: best.pt not found. Check training run folder.")
