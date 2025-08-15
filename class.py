from ultralytics import YOLO

# === 1️⃣ Load a YOLOv8 classification model (pretrained on ImageNet) ===
# "n" = nano (smallest), "s" = small, "m" = medium
model = YOLO("yolov8n-cls.pt")  # good for small datasets; try yolov8s-cls.pt if GPU is strong

# === 2️⃣ Path to dataset ===
# Your dataset folder should look like:
# rank_classifier/train/2, rank_classifier/train/3, ..., rank_classifier/train/A
# rank_classifier/valid/2, rank_classifier/valid/3, ...
dataset_path = "rank_classifier"  # <-- change to your actual dataset folder

# === 3️⃣ Train ===
model.train(
    data=dataset_path,  # folder path (must have train/ and valid/ subfolders)
    epochs=100,          # increase if needed (e.g., 100 for more accuracy)
    imgsz=64,           # small ranks don't need big resolution
    batch=64,           # adjust depending on GPU RAM
    workers=4,          # adjust for CPU threads
    patience=10,        # early stopping if no improvement
    augment=True        # built-in YOLO augmentations
)

print("\n✅ Training complete! Best model saved at: runs/classify/train/weights/best.pt")
