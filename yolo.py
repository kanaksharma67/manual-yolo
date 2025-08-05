import cv2
import easyocr
import numpy as np
import pyautogui
from ultralytics import YOLO
import re
import os
import time

# ==================== CONFIGURATION ====================
MODEL_PATH = "poker_model.pt"          # Path to save/load trained model
DATASET_YAML = "dataset.yaml"          # Path to dataset configuration
SCREEN_REGION = (100, 100, 1000, 700)  # (x,y,width,height) of poker window
CONFIDENCE_THRESHOLD = 0.5             # Detection confidence threshold
USE_OCR = True                         # Enable/disable OCR processing

# Training configuration
TRAINING_CONFIG = {
    "model": "yolov8n.pt",    # Base model (yolov8n.pt, yolov8s.pt, etc.)
    "epochs": 100,            # Number of training epochs
    "imgsz": 640,             # Image size
    "batch": 8,               # Batch size
    "patience": 10,           # Early stopping patience
    "name": "poker_train",    # Training session name
    "exist_ok": True          # Overwrite existing training
}

# Poker element classes from your dataset
CLASSES = {
    0: 'button_call',
    1: 'button_check',
    2: 'button_fold',
    3: 'button_raise',
    4: 'card_1',       # Player hole card 1
    5: 'card_2',       # Player hole card 2
    6: 'flop_1',       # First flop card
    7: 'flop_2',       # Second flop card
    8: 'flop_3',       # Third flop card
    9: 'increaser',
    10: 'my_bet',
    11: 'my_stack',
    12: 'position_BB', # Big Blind position
    13: 'position_SB', # Small Blind position
    14: 'river_1',     # River card
    15: 'total_pot',
    16: 'turn_1',      # Turn card
    17: 'villian_bet',
    18: 'villian_name',
    19: 'villian_stack'
}

# ==================== MODEL TRAINER ====================
class PokerModelTrainer:
    @staticmethod
    def train_model():
        """Train YOLOv8 poker detection model"""
        print("🚀 Starting model training...")
        print(f"⚙️ Configuration: {TRAINING_CONFIG}")
        
        try:
            # Check if dataset exists
            if not os.path.exists(DATASET_YAML):
                raise FileNotFoundError(f"Dataset config not found at {DATASET_YAML}")
            
            # Start timer
            start_time = time.time()
            
            # Load model and train
            model = YOLO(TRAINING_CONFIG["model"])
            results = model.train(
                data=DATASET_YAML,
                epochs=TRAINING_CONFIG["epochs"],
                imgsz=TRAINING_CONFIG["imgsz"],
                batch=TRAINING_CONFIG["batch"],
                patience=TRAINING_CONFIG["patience"],
                name=TRAINING_CONFIG["name"],
                exist_ok=TRAINING_CONFIG["exist_ok"]
            )
            
            # Save the best model
            model.save(MODEL_PATH)
            training_time = (time.time() - start_time) / 60
            print(f"✅ Training completed in {training_time:.1f} minutes")
            print(f"💾 Model saved to {MODEL_PATH}")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return False

# ==================== POKER DETECTOR ====================
class PokerDetector:
    def __init__(self):
        self.model = None
        self.capture = None
        self.ocr = None
        self.game_state = "PREFLOP"
        self.community_cards = []
        self._initialize()

    def _initialize(self):
        """Initialize detector components"""
        try:
            # Initialize OCR if enabled
            if USE_OCR:
                print("🔍 Initializing OCR engine...")
                self.ocr = PokerOCR()
            
            # Initialize screen capture
            self.capture = PokerScreenCapture(SCREEN_REGION)
            
            # Load or train model
            if os.path.exists(MODEL_PATH):
                print(f"💾 Loading existing model from {MODEL_PATH}")
                self.model = YOLO(MODEL_PATH)
            else:
                print("⏳ No trained model found - starting training...")
                if PokerModelTrainer.train_model():
                    self.model = YOLO(MODEL_PATH)
                else:
                    raise RuntimeError("Model training failed")
                    
            print("🎮 Poker detector ready!")
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            self.model = None

    # ... [rest of your existing PokerDetector methods] ...



    def _train_model(self):
        """Train YOLOv8 model if no pretrained exists"""
        print("⏳ Training new poker detection model...")
        model = YOLO("yolov8n.pt")
        model.train(data="dataset.yaml", epochs=100, imgsz=640, batch=8, name='poker_train')
        model.save(MODEL_PATH)
        print(f"✅ Model saved to {MODEL_PATH}")
        return model

    def _update_game_state(self, detections):
        """Determine current game state based on visible cards"""
        new_cards = []
        for det in detections:
            if det['class'].startswith(('flop_', 'turn_', 'river_')):
                if det['text']:  # Only count cards with successfully read values
                    new_cards.append(det['text'])
        
        if len(new_cards) != len(self.community_cards):
            self.community_cards = new_cards
            if len(new_cards) == 0:
                self.game_state = "PREFLOP"
            elif len(new_cards) == 3:
                self.game_state = "FLOP"
            elif len(new_cards) == 4:
                self.game_state = "TURN"
            elif len(new_cards) == 5:
                self.game_state = "RIVER"

    def run(self):
        """Main detection loop"""
        if self.model is None:
            print("❌ Cannot run: Model not loaded")
            return
        
        cv2.namedWindow("Poker Detection", cv2.WINDOW_NORMAL)
        
        while True:
            frame = self.capture.grab_screen()
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD)
            detected_frame = results[0].plot()
            detections = []

            # Process each detection
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = CLASSES[class_id]
                confidence = float(box.conf)
                region = frame[y1:y2, x1:x2]

                # Get OCR result if applicable
                ocr_text = self.ocr.process_detection(class_name, region)
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'text': ocr_text
                })

                # Display OCR results
                if ocr_text:
                    display_text = f"{class_name.split('_')[-1]}: {ocr_text}"
                    cv2.putText(detected_frame, display_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Update game state
            self._update_game_state(detections)

            # Display game state
            cv2.putText(detected_frame, f"Game State: {self.game_state}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Display community cards if any
            if self.community_cards:
                cards_text = " ".join(self.community_cards)
                cv2.putText(detected_frame, f"Community: {cards_text}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.imshow("Poker Detection", detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

# ==================== MAIN EXECUTION ====================
# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("""
    ██████╗  ██████╗ ██╗  ██╗███████╗██████╗     ██████╗ ███████╗████████╗██████╗ 
    ██╔══██╗██╔═══██╗██║ ██╔╝██╔════╝██╔══██╗    ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗
    ██████╔╝██║   ██║█████╔╝ █████╗  ██████╔╝    ██║  ██║█████╗     ██║   ██████╔╝
    ██╔═══╝ ██║   ██║██╔═██╗ ██╔══╝  ██╔══██╗    ██║  ██║██╔══╝     ██║   ██╔══██╗
    ██║     ╚██████╔╝██║  ██╗███████╗██║  ██║    ██████╔╝███████╗   ██║   ██║  ██║
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═╝
    """)
    
    detector = PokerDetector()
    if detector.model is not None:
        detector.run()
    else:
        print("❌ Failed to initialize poker detector")
