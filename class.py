import os
import shutil
from ultralytics import YOLO

# === 1️⃣ Paths ===
dataset_path = "rank_classifier"
project_base = r"C:\Users\HP\monday\manual-yolo"
project_runs = os.path.join(project_base, "runs")
run_name = "rank_classifier"
run_folder = os.path.join(project_runs, run_name)
final_model_path = os.path.join(project_base, "rank_classifier.pt")

# === 🔄 Remove old run folder if exists ===
if os.path.exists(run_folder):
    shutil.rmtree(run_folder)
    print(f"🗑️ Removed old run folder: {run_folder}")

# === 2️⃣ Load YOLOv8 classification model ===
model = YOLO("yolov8n-cls.pt")

# === 3️⃣ Train ===
model.train(
    data=dataset_path,
    epochs=50,
    imgsz=64,
    batch=64,
    workers=4,
    patience=10,
    project=project_runs,
    name=run_name
)

# === 4️⃣ Copy best.pt to fixed location ===
best_model_path = os.path.join(run_folder, "weights", "best.pt")
if os.path.exists(best_model_path):
    shutil.copy(best_model_path, final_model_path)
    print(f"\n✅ Training complete! Best model copied to: {final_model_path}")
else:
    print(f"\n❌ ERROR: best.pt not found at {best_model_path}")
