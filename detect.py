import os
import time
import json
import cv2
import numpy as np
import mss
import torch
import easyocr
from ultralytics import YOLO
import supervision as sv

# ================= CONFIG =================
MODEL_PATH = "poker_model.pt"
RANK_MODEL_PATH = "rank_classifier.pt"
OUTPUT_FOLDER = "live_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SCREEN_REGION = {"top": 0, "left": 0, "width": 1919, "height": 1199}

model = YOLO(MODEL_PATH)
rank_model = YOLO(RANK_MODEL_PATH)
tracker = sv.ByteTrack()

use_gpu_for_ocr = torch.cuda.is_available()
ocr_reader = easyocr.Reader(['en'], gpu=use_gpu_for_ocr)

JSON_OUTPUT = os.path.join(OUTPUT_FOLDER, "detections.json")
all_detections = []

current_game_id = 1
previous_hero_cards = {"card1_rank": "", "card2_rank": "", "card1_suit": "", "card2_suit": ""}
current_game_state = None
last_game_update_time = 0

bbox_annotator = sv.BoxAnnotator()

VALID_CARD_RANKS = {'A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'}
MAPPING_CORRECTION = {'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8', 'T': '10'}

RANK_CLASSES = {
    'card1_rank', 'card2_rank', 'flop1_rank', 'flop2_rank', 'flop3_rank', 'turn_rank', 'river_rank'
}

SUIT_CLASSES = {
    'card1_suite_club', 'card1_suite_diamond', 'card1_suite_heart', 'card1_suite_spades',
    'card2_suite_club', 'card2_suite_diamond', 'card2_suite_heart', 'card2_suite_spades',
    'flop1_suite_club', 'flop1_suite_diamond', 'flop1_suite_heart', 'flop1_suite_spades',
    'flop2_suite_club', 'flop2_suite_diamond', 'flop2_suite_heart', 'flop2_suite_spades',
    'flop3_suite_club', 'flop3_suite_diamond', 'flop3_suite_heart', 'flop3_suite_spades',
    'turn_suite_club', 'turn_suite_diamond', 'turn_suite_heart', 'turn_suite_spades',
    'river_suite_club', 'river_suite_diamond', 'river_suite_heart', 'river_suite_spades'
}

last_screenshot_time = 0
SCREENSHOT_INTERVAL = 0.5
GAME_UPDATE_INTERVAL = 0.5

# -------------------- HELPERS --------------------
def normalize_rank_text(text: str) -> str:
    """Clean OCR text to a valid rank: A,K,Q,J,10,9..2"""
    if not text:
        return ""
    
    t = text.strip().upper()
    
    # Single char corrections first
    if len(t) == 1 and t in MAPPING_CORRECTION:
        t = MAPPING_CORRECTION[t]
    
    # Replace typical OCR junk
    t = t.replace(" ", "").replace("|", "1").replace("O", "0")
    
    # Map T to 10
    if t == "T":
        t = "10"
    
    # Face cards
    if t in {"A", "K", "Q", "J"}:
        return t
    
    # Only digits
    if t.isdigit():
        # Normalize leading zero mistakes: "0" -> "10"
        if t == "0":
            t = "10"
        # Only allow 2..10
        if t in {"10","9","8","7","6","5","4","3","2"}:
            return t
    
    # Try last-chance mapping
    if len(t) == 1 and t in MAPPING_CORRECTION:
        mapped = MAPPING_CORRECTION[t]
        if mapped == "0":
            mapped = "10"
        if mapped in {"10","9","8","7","6","5","4","3","2"}:
            return mapped
    
    return ""

def safe_crop(frame, x1, y1, x2, y2, pad=6):
    """Safely crop frame with bounds checking"""
    if frame is None:
        return None
        
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1 - pad)))
    x2 = max(0, min(w, int(x2 + pad)))
    y1 = max(0, min(h - 1, int(y1 - pad)))
    y2 = max(0, min(h, int(y2 + pad)))
    
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def classify_card_rank(crop, class_name: str = ""):
    """Rank classifier with relaxed threshold for TURN & RIVER"""
    if crop is None or crop.size == 0:
        return ""
    
    try:
        results = rank_model(crop)[0]
        if hasattr(results, 'probs') and results.probs is not None:
            top_class_id = results.probs.top1
            top_confidence = float(results.probs.top1conf)
            pred = rank_model.names.get(top_class_id, "")
            
            # Much more relaxed confidence for turn/river
            if "turn" in class_name.lower() or "river" in class_name.lower():
                threshold = 0.20  # Very low threshold for turn/river
            else:
                threshold = 0.40  # Lower threshold for other cards too
            
            if top_confidence >= threshold:
                cleaned = normalize_rank_text(pred)
                return cleaned if cleaned in VALID_CARD_RANKS else pred.upper()
        return ""
    except Exception as e:
        print(f"Classification error for {class_name}: {e}")
        return ""

def save_screenshot_if_needed(frame, frame_count, current_time):
    global last_screenshot_time
    if current_time - last_screenshot_time >= SCREENSHOT_INTERVAL:
        screenshot_filename = os.path.join(OUTPUT_FOLDER, f"screenshot_frame_{frame_count}_{int(current_time)}.jpg")
        cv2.imwrite(screenshot_filename, frame)
        print(f"[v0] Screenshot saved: {screenshot_filename}")
        last_screenshot_time = current_time

def enhance_for_ocr(image, enhancement_type="standard"):
    """Enhanced OCR preprocessing"""
    if image is None or image.size == 0:
        return image
    
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if enhancement_type == "card_rank":
            height, width = gray.shape
            gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            return binary
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return enhanced
    except Exception as e:
        print(f"Enhancement error: {e}")
        return image

def ocr_text_only(crop, class_name):
    """OCR pipeline with very low confidence for turn/river"""
    if crop is None:
        return ""
    
    # Much lower confidence threshold for turn and river cards
    if any(x in class_name.lower() for x in ['turn_rank', 'river_rank', 'turn_suite', 'river_suite']):
        min_confidence = 0.15  # Very low threshold
    else:
        min_confidence = 0.35
    
    best_text = ""
    best_confidence = 0.0
    
    try:
        # Pass 1: enhanced
        enhanced_crop = enhance_for_ocr(crop, "card_rank" if class_name in RANK_CLASSES else "standard")
        if enhanced_crop is not None:
            ocr_results = ocr_reader.readtext(enhanced_crop, detail=1, paragraph=False)
            for _, text, conf in ocr_results:
                if conf > best_confidence:
                    best_text = text.strip()
                    best_confidence = float(conf)
        
        # Pass 2: otsu on gray
        if best_confidence < 0.7:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_results = ocr_reader.readtext(thresh, detail=1, paragraph=False)
            for _, text, conf in ocr_results:
                if conf > best_confidence:
                    best_text = text.strip()
                    best_confidence = float(conf)
        
        # Pass 3: scaled up raw
        if best_confidence < 0.6:
            resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ocr_results = ocr_reader.readtext(resized, detail=1, paragraph=False)
            for _, text, conf in ocr_results:
                if conf > best_confidence:
                    best_text = text.strip()
                    best_confidence = float(conf)
        
        if class_name in RANK_CLASSES:
            best_text = normalize_rank_text(best_text)
        
        return best_text if best_confidence > min_confidence else ""
    except Exception as e:
        print(f"OCR error for {class_name}: {e}")
        return ""

def extract_text_with_multiple_methods(crop, class_name):
    """Unified text extractor with better turn/river handling"""
    if crop is None:
        return ""
    
    # Suits handled via class name
    if class_name.lower() in {name.lower() for name in SUIT_CLASSES}:
        return ""
    
    if class_name in RANK_CLASSES:
        # 1) Classifier first
        text = classify_card_rank(crop, class_name)
        # 2) Fallback to OCR if empty
        if not text:
            text = ocr_text_only(crop, class_name)
        # 3) Final normalization and validation
        text = normalize_rank_text(text)
        return text if text in VALID_CARD_RANKS else text
    
    # Non-rank generic OCR
    return ocr_text_only(crop, class_name)

def create_clean_detections(xyxy, class_id=None, confidence=None, tracker_id=None):
    """Create clean detections with proper None handling"""
    if len(xyxy) == 0:
        return sv.Detections.empty()
    
    # Ensure xyxy is proper numpy array
    xyxy = np.array(xyxy, dtype=np.float32)
    
    # Handle class_id
    if class_id is None:
        class_id = np.zeros(len(xyxy), dtype=np.int32)
    else:
        clean_class_ids = []
        for cid in class_id:
            try:
                if cid is None or np.isnan(float(cid)):
                    clean_class_ids.append(0)
                else:
                    clean_class_ids.append(int(cid))
            except (ValueError, TypeError):
                clean_class_ids.append(0)
        class_id = np.array(clean_class_ids, dtype=np.int32)
    
    # Handle confidence
    if confidence is None:
        confidence = np.ones(len(xyxy), dtype=np.float32)
    else:
        clean_conf = []
        for c in confidence:
            try:
                if c is None or np.isnan(float(c)):
                    clean_conf.append(0.0)
                else:
                    clean_conf.append(float(c))
            except (ValueError, TypeError):
                clean_conf.append(0.0)
        confidence = np.array(clean_conf, dtype=np.float32)
    
    # Handle tracker_id
    if tracker_id is not None:
        clean_tracker_ids = []
        for tid in tracker_id:
            try:
                if tid is None or np.isnan(float(tid)):
                    clean_tracker_ids.append(-1)
                else:
                    clean_tracker_ids.append(int(tid))
            except (ValueError, TypeError):
                clean_tracker_ids.append(-1)
        tracker_id = np.array(clean_tracker_ids, dtype=np.int32)
    
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        tracker_id=tracker_id
    )
    return detections

def determine_game_state(detections_data):
    """Determine current game state based on board cards"""
    flop_cards = 0
    turn_card = False
    river_card = False
    
    for detection in detections_data:
        class_name = detection.get("class_name", "")
        ocr_text = detection.get("ocr_text", "")
        
        if "flop" in class_name and "rank" in class_name and ocr_text:
            flop_cards += 1
        elif "turn" in class_name and "rank" in class_name and ocr_text:
            turn_card = True
        elif "river" in class_name and "rank" in class_name and ocr_text:
            river_card = True
    
    if river_card:
        return "river"
    elif turn_card:
        return "turn"
    elif flop_cards >= 3:
        return "flop"
    else:
        return "preflop"

def check_for_new_game(current_hero_cards, previous_hero_cards):
    """Check if new hole cards indicate a new game"""
    # If no previous cards, it's a new game
    if not previous_hero_cards["card1_rank"] and not previous_hero_cards["card2_rank"]:
        return True
    
    # Check if any card changed
    card1_changed = (current_hero_cards["card1_rank"] and 
                    current_hero_cards["card1_rank"] != previous_hero_cards["card1_rank"])
    card2_changed = (current_hero_cards["card2_rank"] and 
                    current_hero_cards["card2_rank"] != previous_hero_cards["card2_rank"])
    card1_suit_changed = (current_hero_cards["card1_suit"] and 
                         current_hero_cards["card1_suit"] != previous_hero_cards["card1_suit"])
    card2_suit_changed = (current_hero_cards["card2_suit"] and 
                         current_hero_cards["card2_suit"] != previous_hero_cards["card2_suit"])
    
    return card1_changed or card2_changed or card1_suit_changed or card2_suit_changed

def get_suit_name(class_name):
    """Convert class name to suit name - fixed inconsistency"""
    cl = class_name.lower()
    if "club" in cl:
        return "of club"
    elif "diamond" in cl:
        return "of diamond"
    elif "heart" in cl:
        return "of heart"
    elif "spade" in cl:  # Fixed: both "spade" and "spades" map to "of spade"
        return "of spade"
    return ""

def update_game_data(game_state, detections):
    """Update game state with detection data - improved turn/river handling"""
    for detection in detections:
        class_name = detection.get("class_name", "")
        ocr_text = detection.get("ocr_text", "")
        bbox = detection.get("bbox", [])
        
        # Debug print for turn/river
        if "turn" in class_name or "river" in class_name:
            print(f"[DEBUG] Processing {class_name}: '{ocr_text}'")
        
        # Handle hero cards
        if class_name == "card1_rank" and ocr_text:
            game_state["hero"]["cards"][0]["rank"] = ocr_text
        elif class_name == "card2_rank" and ocr_text:
            game_state["hero"]["cards"][1]["rank"] = ocr_text
        elif class_name in ["card1_suite_club", "card1_suite_diamond", "card1_suite_heart", "card1_suite_spades"]:
            game_state["hero"]["cards"][0]["suit"] = get_suit_name(class_name)
        elif class_name in ["card2_suite_club", "card2_suite_diamond", "card2_suite_heart", "card2_suite_spades"]:
            game_state["hero"]["cards"][1]["suit"] = get_suit_name(class_name)
        
        # Handle board cards - FIXED LOGIC
        elif class_name == "flop1_rank" and ocr_text:
            game_state["board"]["flop"][0]["rank"] = ocr_text
        elif class_name == "flop2_rank" and ocr_text:
            game_state["board"]["flop"][1]["rank"] = ocr_text
        elif class_name == "flop3_rank" and ocr_text:
            game_state["board"]["flop"][2]["rank"] = ocr_text
        elif class_name == "turn_rank" and ocr_text:
            game_state["board"]["turn"]["rank"] = ocr_text
            print(f"[DEBUG] Set turn rank: {ocr_text}")
        elif class_name == "river_rank" and ocr_text:
            game_state["board"]["river"]["rank"] = ocr_text
            print(f"[DEBUG] Set river rank: {ocr_text}")
        
        # Handle board suits - FIXED LOGIC
        elif class_name in ["flop1_suite_club", "flop1_suite_diamond", "flop1_suite_heart", "flop1_suite_spades"]:
            game_state["board"]["flop"][0]["suit"] = get_suit_name(class_name)
        elif class_name in ["flop2_suite_club", "flop2_suite_diamond", "flop2_suite_heart", "flop2_suite_spades"]:
            game_state["board"]["flop"][1]["suit"] = get_suit_name(class_name)
        elif class_name in ["flop3_suite_club", "flop3_suite_diamond", "flop3_suite_heart", "flop3_suite_spades"]:
            game_state["board"]["flop"][2]["suit"] = get_suit_name(class_name)
        elif class_name in ["turn_suite_club", "turn_suite_diamond", "turn_suite_heart", "turn_suite_spades"]:
            game_state["board"]["turn"]["suit"] = get_suit_name(class_name)
            print(f"[DEBUG] Set turn suit: {get_suit_name(class_name)}")
        elif class_name in ["river_suite_club", "river_suite_diamond", "river_suite_heart", "river_suite_spades"]:
            game_state["board"]["river"]["suit"] = get_suit_name(class_name)
            print(f"[DEBUG] Set river suit: {get_suit_name(class_name)}")
        
        # Handle villains
        elif class_name.startswith("villian") and "_name" in class_name:
            villain_num = class_name[7]  # Get the number after "villian"
            found = False
            for v in game_state["villains"]:
                if v["position"] == villain_num:
                    v["name"] = ocr_text
                    found = True
                    break
            if not found:
                game_state["villains"].append({
                    "position": villain_num,
                    "name": ocr_text,
                    "stack": "",
                    "bet": ""
                })
        elif class_name.startswith("villian") and "_stack" in class_name:
            villain_num = class_name[7]
            for v in game_state["villains"]:
                if v["position"] == villain_num:
                    v["stack"] = ocr_text
                    break
        elif class_name.startswith("villian") and "_bet" in class_name:
            villain_num = class_name[7]
            for v in game_state["villains"]:
                if v["position"] == villain_num:
                    v["bet"] = ocr_text
                    break
        
        # Handle hero stack/bet
        elif class_name == "my_stack":
            game_state["hero"]["stack"] = ocr_text
        elif class_name == "my_bet":
            game_state["hero"]["bet"] = ocr_text
        
        # Handle pot
        elif class_name == "total_pot":
            game_state["pot"] = ocr_text
        
        # Handle buttons
        elif class_name == "button_fold":
            game_state["ui"]["buttons"]["fold"] = {"coordinates": bbox, "text": ocr_text}
        elif class_name == "button_check":
            game_state["ui"]["buttons"]["check"] = {"coordinates": bbox, "text": ocr_text}
        elif class_name == "button_call":
            game_state["ui"]["buttons"]["call"] = {"coordinates": bbox, "text": ocr_text}
        elif class_name == "button_raise":
            game_state["ui"]["buttons"]["raise"] = {"coordinates": bbox, "text": ocr_text}
        elif class_name == "button_bet":
            game_state["ui"]["buttons"]["bet"] = {"coordinates": bbox, "text": ocr_text}
        elif class_name == "button_allin":
            game_state["ui"]["buttons"]["allin"] = {"coordinates": bbox, "text": ocr_text}
        elif class_name == "iinput_field":
            game_state["ui"]["bet_input"] = {"coordinates": bbox, "text": ocr_text}
    
    # Update game state
    game_state["game_state"] = determine_game_state(detections)

def save_game_data(game_id, game_state):
    """Save game data to JSON file"""
    game_filename = os.path.join(OUTPUT_FOLDER, f"game_{game_id}.json")
    try:
        with open(game_filename, "w", encoding="utf-8") as f:
            json.dump(game_state, f, indent=2)
        print(f"Updated game {game_id} data in {game_filename}")
    except Exception as e:
        print(f"Error saving game {game_id}: {e}")

def initialize_game_state():
    """Initialize a new game state structure"""
    return {
        "game_id": current_game_id,
        "game_state": "preflop",
        "villains": [],
        "hero": {
            "stack": "",
            "bet": "",
            "cards": [
                {"rank": "", "suit": ""},
                {"rank": "", "suit": ""}
            ]
        },
        "board": {
            "flop": [
                {"rank": "", "suit": ""},
                {"rank": "", "suit": ""},
                {"rank": "", "suit": ""}
            ],
            "turn": {"rank": "", "suit": ""},
            "river": {"rank": "", "suit": ""}
        },
        "pot": "",
        "ui": {
            "buttons": {
                "fold": {"coordinates": [], "text": ""},
                "check": {"coordinates": [], "text": ""},
                "call": {"coordinates": [], "text": ""},
                "raise": {"coordinates": [], "text": ""},
                "bet": {"coordinates": [], "text": ""},
                "allin": {"coordinates": [], "text": ""}
            },
            "bet_input": {"coordinates": [], "text": ""}
        }
    }

# Initialize current game state
current_game_state = initialize_game_state()

# ================= MAIN LOOP =================
with mss.mss() as sct:
    frame_count = 0
    print("Starting live detection. Press 'q' in the window to quit.")
    
    while True:
        start_time = time.time()
        
        # Capture screen
        screenshot = np.array(sct.grab(SCREEN_REGION))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        save_screenshot_if_needed(frame, frame_count, start_time)
        
        # YOLO detection
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Skip if no detections
        if len(detections.xyxy) == 0:
            cv2.imshow("Live Detection + OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1
            continue
        
        # Create clean detections object
        detections = create_clean_detections(
            detections.xyxy,
            detections.class_id,
            detections.confidence
        )
        
        # Track objects
        try:
            tracked = tracker.update_with_detections(detections)
        except Exception as e:
            print(f"Tracking error: {e}")
            tracked = detections
        
        dets_for_annotate = None
        labels = []
        frame_data = []
        
        # Process tracked detections
        if isinstance(tracked, sv.Detections) and len(tracked.xyxy) > 0:
            dets_for_annotate = create_clean_detections(
                tracked.xyxy,
                tracked.class_id,
                tracked.confidence,
                tracked.tracker_id
            )
            
            # Process each detection
            for i in range(len(dets_for_annotate.xyxy)):
                x1, y1, x2, y2 = map(int, dets_for_annotate.xyxy[i])
                class_id = int(dets_for_annotate.class_id[i])
                tracker_id = int(dets_for_annotate.tracker_id[i]) if dets_for_annotate.tracker_id is not None else -1
                conf = float(dets_for_annotate.confidence[i])
                
                crop = safe_crop(frame, x1, y1, x2, y2, pad=6)
                class_name = model.names.get(class_id, f"class{class_id}")
                ocr_text = extract_text_with_multiple_methods(crop, class_name)
                
                frame_data.append({
                    "frame": frame_count,
                    "tracker_id": tracker_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 3),
                    "ocr_text": ocr_text
                })
                
                labels.append(f"ID{tracker_id} {class_name} {ocr_text}")
        
        # If tracker didn't return detections, use original
        if dets_for_annotate is None or len(dets_for_annotate.xyxy) == 0:
            dets_for_annotate = detections
            
            for i in range(len(dets_for_annotate.xyxy)):
                x1, y1, x2, y2 = map(int, dets_for_annotate.xyxy[i])
                class_id = int(dets_for_annotate.class_id[i])
                conf = float(dets_for_annotate.confidence[i])
                
                crop = safe_crop(frame, x1, y1, x2, y2, pad=6)
                class_name = model.names.get(class_id, f"class{class_id}")
                ocr_text = extract_text_with_multiple_methods(crop, class_name)
                
                frame_data.append({
                    "frame": frame_count,
                    "tracker_id": -1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 3),
                    "ocr_text": ocr_text
                })
                
                labels.append(f"{class_name} {ocr_text}")
        
        # Extract hero cards for new game detection
        detected_hero_cards = {"card1_rank": "", "card2_rank": "", "card1_suit": "", "card2_suit": ""}
        for detection in frame_data:
            if detection["class_name"] == "card1_rank" and detection["ocr_text"]:
                detected_hero_cards["card1_rank"] = detection["ocr_text"]
            elif detection["class_name"] == "card2_rank" and detection["ocr_text"]:
                detected_hero_cards["card2_rank"] = detection["ocr_text"]
            elif detection["class_name"] in ["card1_suite_club", "card1_suite_diamond", "card1_suite_heart", "card1_suite_spades"]:
                detected_hero_cards["card1_suit"] = get_suit_name(detection["class_name"])
            elif detection["class_name"] in ["card2_suite_club", "card2_suite_diamond", "card2_suite_heart", "card2_suite_spades"]:
                detected_hero_cards["card2_suit"] = get_suit_name(detection["class_name"])
        
        # Check for new game
        if check_for_new_game(detected_hero_cards, previous_hero_cards):
            # Save previous game if it had cards
            if (current_game_state["hero"]["cards"][0]["rank"] or 
                current_game_state["hero"]["cards"][1]["rank"]):
                print(f"[v0] Saving game {current_game_id} before starting new game")
                save_game_data(current_game_id, current_game_state)
                current_game_id += 1
            
            previous_hero_cards = detected_hero_cards.copy()
            current_game_state = initialize_game_state()
            current_game_state["game_id"] = current_game_id
            print(f"[v0] New game detected! Game ID: {current_game_id}, Cards: {detected_hero_cards}")
        
        # Update current game state with new detections
        update_game_data(current_game_state, frame_data)
        
        # Save game state periodically
        if time.time() - last_game_update_time >= GAME_UPDATE_INTERVAL:
            save_game_data(current_game_id, current_game_state)
            last_game_update_time = time.time()
        
        # Annotate frame
        if len(dets_for_annotate.xyxy) > 0:
            # Ensure labels match detections
            num_boxes = len(dets_for_annotate.xyxy)
            if len(labels) < num_boxes:
                labels += [""] * (num_boxes - len(labels))
            elif len(labels) > num_boxes:
                labels = labels[:num_boxes]
            
            try:
                annotated_frame = bbox_annotator.annotate(frame.copy(), dets_for_annotate, labels)
            except Exception as e:
                print(f"Annotation error: {e}")
                annotated_frame = frame.copy()
        else:
            annotated_frame = frame.copy()
        
        # Save all detections to JSON
        all_detections.append({
            "frame": frame_count,
            "timestamp": time.time(),
            "detections": frame_data
        })
        
        # Write detections JSON
        try:
            with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
                json.dump(all_detections, f, indent=2)
        except Exception as e:
            print(f"Error writing detections JSON: {e}")
        
        # Display frame
        cv2.imshow("Live Detection + OCR", annotated_frame)
        
        frame_count += 1
        fps = 1.0 / (time.time() - start_time + 1e-6)
        print(f"Frame {frame_count} | FPS: {fps:.2f} | Detections: {len(frame_data)} | Game: {current_game_id}")
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Save final game state
if (current_game_state["hero"]["cards"][0]["rank"] or 
    current_game_state["hero"]["cards"][1]["rank"]):
    print(f"[v0] Saving final game {current_game_id}")
    save_game_data(current_game_id, current_game_state)

cv2.destroyAllWindows()
