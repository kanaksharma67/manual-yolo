import cv2
import easyocr
import numpy as np
import pyautogui
from ultralytics import YOLO
import re
import os
import time

# ==================== CONFIGURATION ====================
MODEL_PATH = "poker_model.pt"
DATASET_YAML = "data.yaml"
SCREEN_REGION = (100, 100, 1000, 700)
CONFIDENCE_THRESHOLD = 0.5
USE_OCR = True

# Training configuration
TRAINING_CONFIG = {
    "model": "yolov8m.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "patience": 10,
    "name": "poker_train",
    "exist_ok": True
}

CLASSES = {
    0: 'button_call',
    1: 'button_check',
    2: 'button_fold',
    3: 'button_raise',
    4: 'card_1',
    5: 'card_2',
    6: 'flop_1',
    7: 'flop_2',
    8: 'flop_3',
    9: 'increaser',
    10: 'my_bet',
    11: 'my_stack',
    12: 'position_BB',
    13: 'position_SB',
    14: 'river_1',
    15: 'total_pot',
    16: 'turn_1',
    17: 'villian_bet',
    18: 'villian_name',
    19: 'villian_stack'
}

# ==================== OCR PROCESSOR ====================
class PokerOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en']) if USE_OCR else None
        self.card_pattern = re.compile(r'^([AKQJT2-9]|10)[shdc♠♥♦♣]$', re.IGNORECASE)
        self.numeric_pattern = re.compile(r'[\d,.]+[kKmMbB]?')
        self.pot_pattern = re.compile(r'pot[:]?\s*([\d,.kKbBmM]+)', re.IGNORECASE)
        self.name_pattern = re.compile(r'^[a-zA-Z0-9_]+$')

    def process_detection(self, class_name, region):
        """Process detected region based on its class"""
        if not USE_OCR or self.reader is None:
            return None

        try:
            if class_name.startswith(('card_', 'flop_', 'turn_', 'river_')):
                return self._extract_card_value(region)
            elif class_name in ('my_bet', 'villian_bet', 'my_stack', 'villian_stack', 'increaser'):
                return self._extract_numeric_value(region)
            elif class_name == 'total_pot':
                return self._extract_pot_value(region)
            elif class_name in ('villian_name',):
                return self._extract_name(region)
            elif class_name.startswith('button_'):
                return self._extract_button_text(region)
            elif class_name.startswith('position_'):
                return class_name.split('_')[-1]
        except Exception as e:
            print(f"OCR Error for {class_name}: {str(e)}")
        return None

    def _extract_card_value(self, region):
        """Extract poker card value (Ah, Ks, etc.)"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        result = self.reader.readtext(binary, allowlist='AKQJT2345678910shdc♠♥♦♣', detail=0)
        if result:
            text = ''.join(result).upper()
            text = text.replace('10', 'T').replace('♠', 's').replace('♥', 'h').replace('♦', 'd').replace('♣', 'c')
            if self.card_pattern.match(text):
                return text
        return None

    def _extract_numeric_value(self, region):
        """Extract numeric values like bets and stacks"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        result = self.reader.readtext(enhanced, allowlist='0123456789.,kKmMbB$', detail=0)
        if result:
            text = ''.join(result).upper()
            match = self.numeric_pattern.search(text)
            return match.group() if match else None
        return None

    def _extract_pot_value(self, region):
        """Special handling for pot values"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        result = self.reader.readtext(gray, allowlist='0123456789.,kKmMbB$pPoOtT', detail=0)
        if result:
            text = ''.join(result).upper()
            match = self.pot_pattern.search(text)
            return match.group(1) if match else None
        return None

    def _extract_name(self, region):
        """Extract player names"""
        result = self.reader.readtext(region, detail=0)
        if result and len(result[0]) > 1:
            text = result[0]
            if self.name_pattern.match(text):
                return text
        return None

    def _extract_button_text(self, region):
        """Confirm button text matches expected action"""
        result = self.reader.readtext(region, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=0)
        if result:
            return result[0].upper()
        return None

# ==================== SCREEN CAPTURE ====================
class PokerScreenCapture:
    def __init__(self, region):
        self.region = region
        
    def grab_screen(self):
        screenshot = pyautogui.screenshot(region=self.region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# ==================== MODEL TRAINER ====================
class PokerModelTrainer:
    @staticmethod
    def train_model():
        """Train YOLOv8 poker detection model"""
        print("🚀 Starting model training...")
        print(f"⚙️ Configuration: {TRAINING_CONFIG}")
        
        try:
            if not os.path.exists(DATASET_YAML):
                raise FileNotFoundError(f"Dataset config not found at {DATASET_YAML}")
            
            start_time = time.time()
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
            if USE_OCR:
                print("🔍 Initializing OCR engine...")
                self.ocr = PokerOCR()
            
            self.capture = PokerScreenCapture(SCREEN_REGION)
            
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

    def _update_game_state(self, detections):
        """Determine current game state based on visible cards"""
        new_cards = []
        for det in detections:
            if det['class'].startswith(('flop_', 'turn_', 'river_')):
                if det['text']:
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

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = CLASSES[class_id]
                confidence = float(box.conf)
                region = frame[y1:y2, x1:x2]

                ocr_text = self.ocr.process_detection(class_name, region) if self.ocr else None
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'text': ocr_text
                })

                if ocr_text:
                    display_text = f"{class_name.split('_')[-1]}: {ocr_text}"
                    cv2.putText(detected_frame, display_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            self._update_game_state(detections)

            cv2.putText(detected_frame, f"Game State: {self.game_state}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if self.community_cards:
                cards_text = " ".join(self.community_cards)
                cv2.putText(detected_frame, f"Community: {cards_text}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.imshow("Poker Detection", detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

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
