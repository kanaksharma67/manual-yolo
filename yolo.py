import cv2
import easyocr
import numpy as np
import pyautogui
from ultralytics import YOLO
import re
import os
import time
from collections import Counter
import yaml

# ==================== CONFIGURATION ====================
MODEL_PATH = "poker_model.pt"
DATASET_YAML = "roadmap1.v3i.yolov8/data.yaml"  
SCREEN_REGION = (100, 100, 1000, 700)
CONFIDENCE_THRESHOLD = 0.5
USE_OCR = True

# Enhanced Training Configuration
TRAINING_CONFIG = {
    "model": "yolov8m.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "patience": 15,
    "name": "poker_train",
    "exist_ok": True,
    "dropout": 0.2,
    "lr0": 0.01,
    "lrf": 0.001,
    "momentum": 0.98,
    "weight_decay": 0.0001,
    "warmup_epochs": 5,
    "optimizer": "AdamW",
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 15,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0005,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.2,
    "copy_paste": 0.5
}

CLASSES = {i: name for i, name in enumerate([
    'button_allin', 'button_bet', 'button_call', 'button_check', 'button_fold',
    'button_raise', 'card1_rank', 'card1_suite_club', 'card1_suite_diamond',
    'card1_suite_heart', 'card1_suite_spades', 'card2_rank', 'card2_suite_club',
    'card2_suite_diamond', 'card2_suite_heart', 'card2_suite_spades',
    'flop1_rank', 'flop1_suite_club', 'flop1_suite_diamond', 'flop1_suite_heart',
    'flop1_suite_spades', 'flop2_rank', 'flop2_suite_club', 'flop2_suite_diamond',
    'flop2_suite_heart', 'flop2_suite_spades', 'flop3_rank', 'flop3_suite_club',
    'flop3_suite_diamond', 'flop3_suite_heart', 'flop3_suite_spades', 'game_id',
    'iinput_field', 'my_bet', 'my_stack', 'position_BB', 'position_SB',
    'river_rank', 'river_suite_club', 'river_suite_diamond', 'river_suite_heart',
    'river_suite_spades', 'total_pot', 'turn_rank', 'turn_suite_club',
    'turn_suite_diamond', 'turn_suite_heart', 'turn_suite_spades',
    'villian1_bet', 'villian1_name', 'villian1_stack', 'villian2_bet',
    'villian2_name', 'villian2_stack', 'villian3_bet', 'villian3_name',
    'villian3_stack', 'villian4_bet', 'villian4_name', 'villian4_stack',
    'villian5_bet', 'villian5_name', 'villian5_stack', 'winner'
])}

# ==================== ENHANCED OCR PROCESSOR ====================
class PokerOCR:
    def __init__(self):
        if USE_OCR:
            print("🔥 Initializing Enhanced OCR Engine...")
            self.reader = easyocr.Reader(
                ['en'],
                gpu=True,
                model_storage_directory='custom_ocr_models',
                download_enabled=True
            )
        else:
            self.reader = None
            
        self.card_pattern = re.compile(r'^([AKQJT2-9]|10)[shdc♠♥♦♣]$', re.IGNORECASE)
        self.numeric_pattern = re.compile(r'[\d,.]+[kKmMbB]?')
        self.pot_pattern = re.compile(r'pot[:]?\s*([\d,.kKbBmM]+)', re.IGNORECASE)
        self.name_pattern = re.compile(r'^[a-zA-Z0-9_]{3,20}$')

    def process_detection(self, class_name, region):
        """Enhanced OCR processing with poker-specific rules"""
        if not USE_OCR or self.reader is None:
            return None

        try:
            # Card detection
            if class_name.startswith(('card_', 'flop_', 'turn_', 'river_')):
                return self._extract_card_value(region)
            
            # Numeric values
            elif class_name in ('my_bet', 'villian_bet', 'my_stack', 'villian_stack', 'increaser'):
                return self._extract_numeric_value(region)
            
            # Pot values
            elif class_name == 'total_pot':
                return self._extract_pot_value(region)
                
            # Player names
            elif class_name in ('villian_name',):
                return self._extract_name(region)
                
        except Exception as e:
            print(f"⚠️ OCR Error for {class_name}: {str(e)}")
        return None

    def _preprocess_region(self, region, is_card=False):
        """Enhanced image preprocessing"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        if is_card:
            # Special processing for cards
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
            processed = clahe.apply(gray)
            processed = cv2.GaussianBlur(processed, (3,3), 0)
        else:
            # For numbers/text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            processed = clahe.apply(gray)
            
        return processed

    def _extract_card_value(self, region):
        """Advanced card recognition with validation"""
        processed = self._preprocess_region(region, is_card=True)
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Try multiple OCR approaches
        for scale in [1.0, 1.5, 2.0]:
            scaled = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            result = self.reader.readtext(
                scaled,
                allowlist='AKQJT2345678910shdc♠♥♦♣',
                detail=0,
                paragraph=False,
                min_size=20
            )
            
            if result:
                text = ''.join(result).upper()
                text = text.replace('10', 'T').replace('♠', 's').replace('♥', 'h').replace('♦', 'd').replace('♣', 'c')
                if self.card_pattern.match(text):
                    return text
                    
        return None

    def _extract_numeric_value(self, region):
        """Precision numeric extraction"""
        processed = self._preprocess_region(region)
        result = self.reader.readtext(
            processed,
            allowlist='0123456789.,kKmMbB$',
            detail=0,
            paragraph=False
        )
        
        if result:
            text = ''.join(result).upper()
            match = self.numeric_pattern.search(text)
            return match.group() if match else None
        return None

    def _extract_pot_value(self, region):
        """Specialized pot value reading"""
        processed = self._preprocess_region(region)
        result = self.reader.readtext(
            processed,
            allowlist='0123456789.,kKmMbB$pPoOtT',
            detail=0,
            paragraph=False
        )
        
        if result:
            text = ''.join(result).upper()
            match = self.pot_pattern.search(text)
            return match.group(1) if match else None
        return None

    def _extract_name(self, region):
        """Player name validation"""
        processed = self._preprocess_region(region)
        result = self.reader.readtext(
            processed,
            allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_',
            detail=0,
            paragraph=False
        )
        
        if result and len(result[0]) >= 3:
            text = result[0]
            if self.name_pattern.match(text):
                return text
        return None

# ==================== SCREEN CAPTURE ====================
class PokerScreenCapture:
    def __init__(self, region):
        self.region = region
        
    def grab_screen(self):
        screenshot = pyautogui.screenshot(region=self.region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# ==================== ENHANCED MODEL TRAINER ====================
class PokerModelTrainer:
    @staticmethod
    def train_model():
        """Advanced training with validation checks"""
        print("🚀 Starting Advanced Model Training...")
        print(f"⚙️ Configuration: {TRAINING_CONFIG}")
        
        try:
            # Verify dataset
            if not os.path.exists(DATASET_YAML):
                raise FileNotFoundError(f"Dataset config not found at {DATASET_YAML}")
                
            # Check class balance
            PokerModelTrainer._verify_dataset(DATASET_YAML)
            
            # Start training
            start_time = time.time()
            model = YOLO(TRAINING_CONFIG["model"])
            
            # Train with augmented configuration
            results = model.train(
                data=DATASET_YAML,
                epochs=TRAINING_CONFIG["epochs"],
                imgsz=TRAINING_CONFIG["imgsz"],
                batch=TRAINING_CONFIG["batch"],
                patience=TRAINING_CONFIG["patience"],
                name=TRAINING_CONFIG["name"],
                exist_ok=TRAINING_CONFIG["exist_ok"],
                # Architecture
                
                dropout=TRAINING_CONFIG["dropout"],
                # Optimization
                lr0=TRAINING_CONFIG["lr0"],
                lrf=TRAINING_CONFIG["lrf"],
                momentum=TRAINING_CONFIG["momentum"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                warmup_epochs=TRAINING_CONFIG["warmup_epochs"],
                optimizer=TRAINING_CONFIG["optimizer"],
                # Augmentation
                hsv_h=TRAINING_CONFIG["hsv_h"],
                hsv_s=TRAINING_CONFIG["hsv_s"],
                hsv_v=TRAINING_CONFIG["hsv_v"],
                degrees=TRAINING_CONFIG["degrees"],
                translate=TRAINING_CONFIG["translate"],
                scale=TRAINING_CONFIG["scale"],
                shear=TRAINING_CONFIG["shear"],
                perspective=TRAINING_CONFIG["perspective"],
                flipud=TRAINING_CONFIG["flipud"],
                fliplr=TRAINING_CONFIG["fliplr"],
                mosaic=TRAINING_CONFIG["mosaic"],
                mixup=TRAINING_CONFIG["mixup"],
                copy_paste=TRAINING_CONFIG["copy_paste"]
            )
            
            # Save and validate
            model.save(MODEL_PATH)
            training_time = (time.time() - start_time) / 60
            
            # Run validation
            metrics = model.val()
            print(f"\n✅ Training completed in {training_time:.1f} minutes")
            print(f"💾 Model saved to {MODEL_PATH}")
            print(f"\n📊 Validation Results:")
            print(f"mAP50: {metrics.box.map50:.1%}")
            print(f"mAP50-95: {metrics.box.map:.1%}")
            print(f"Precision: {metrics.box.p:.1%}")
            print(f"Recall: {metrics.box.r:.1%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            return False

    @staticmethod
    def _verify_dataset(yaml_path):
        """Check dataset quality before training"""
        print("🔍 Verifying dataset balance...")
        
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        label_dir = os.path.join(os.path.dirname(yaml_path), data['train'].replace('/images', '/labels'))
        class_counts = Counter()
        
        for label_file in os.listdir(label_dir):
            with open(os.path.join(label_dir, label_file)) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    
        print("\n📊 Class Distribution:")
        for class_id, count in class_counts.items():
            print(f"{CLASSES[class_id]}: {count} samples")
            
        if min(class_counts.values()) < 10:
            print("\n⚠️ Warning: Some classes have very few samples!")
            print("Recommend at least 50 samples per class for good accuracy")

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
        """Initialize with enhanced checks"""
        try:
            # Initialize OCR
            if USE_OCR:
                self.ocr = PokerOCR()
            
            # Initialize capture
            self.capture = PokerScreenCapture(SCREEN_REGION)
            
            # Load or train model
            if os.path.exists(MODEL_PATH):
                print(f"💾 Loading model from {MODEL_PATH}")
                self.model = YOLO(MODEL_PATH)
                
                # Verify model
                if not hasattr(self.model, 'names'):
                    raise RuntimeError("Invalid model format")
            else:
                print("⏳ No model found - starting training...")
                if PokerModelTrainer.train_model():
                    self.model = YOLO(MODEL_PATH)
                else:
                    raise RuntimeError("Training failed")
                    
            print("🎮 Poker detector ready!")
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            self.model = None

    def _update_game_state(self, detections):
        """Enhanced game state tracking"""
        new_cards = []
        for det in detections:
            if det['class'].startswith(('flop_', 'turn_', 'river_')):
                if det['text'] and self._validate_card(det['text']):
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

    def _validate_card(self, card_text):
        """Strict card validation"""
        if not card_text or len(card_text) < 2:
            return False
        rank, suit = card_text[:-1], card_text[-1]
        return (rank in "AKQJT98765432") and (suit in "shdc")

    def run(self):
        """Enhanced detection loop"""
        if self.model is None:
            print("❌ Cannot run: Model not loaded")
            return
        
        cv2.namedWindow("Poker Detection", cv2.WINDOW_NORMAL)
        
        while True:
            frame = self.capture.grab_screen()
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD)
            detected_frame = results[0].plot()
            detections = []

            # Process detections with enhanced OCR
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = CLASSES[class_id]
                confidence = float(box.conf)
                region = frame[y1:y2, x1:x2]

                ocr_text = self.ocr.process_detection(class_name, region) if self.ocr else None
                
                # Apply poker-specific validation
                if class_name.startswith(('card_', 'flop_', 'turn_', 'river_')) and ocr_text:
                    if not self._validate_card(ocr_text):
                        ocr_text = None
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'text': ocr_text
                })

                # Display with enhanced formatting
                if ocr_text:
                    color = (0, 255, 0)  # Green for valid
                    if class_name.startswith(('card_', 'flop_', 'turn_', 'river_')):
                        color = (255, 0, 255)  # Purple for cards
                    
                    display_text = f"{class_name.split('_')[-1]}: {ocr_text} ({confidence:.0%})"
                    cv2.putText(detected_frame, display_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update game state
            self._update_game_state(detections)

            # Enhanced display
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