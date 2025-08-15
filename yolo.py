# updated_poker_detector.py
import os
import re
import time
import json
import tempfile
import base64
from math import ceil, sqrt
from typing import List, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image

import pyautogui
from ultralytics import YOLO
import re
import os
import time
from collections import Counter
import yaml
import json


# ==================== CONFIGURATION ====================

MODEL_PATH = "poker_model.pt"
DATASET_YAML = "roadmap1.v3i.yolov8/data.yaml"

FULLSCREEN = True
SCREEN_REGION = (100, 100, 1000, 700)

CONFIDENCE_THRESHOLD = 0.25
USE_OCR = True
DEBUG = False
LOOP_INTERVAL_SECONDS = 2

# Use GPT-4o for better accuracy when local OCR fails
USE_GPT_FALLBACK = True
GPT_IMAGE_MODEL = "gpt-4o"  # Using GPT-4o for better image analysis

# Enhanced detection settings
CROP_INDIVIDUAL_CARDS = False     # Temporarily disable
CROP_COMMUNITY_CARDS = False      # Temporarily disable
CREATE_DETECTION_COLLAGE = False  # Temporarily disable
ACCUMULATE_DATA = False           # Temporarily disable
SAVE_INDIVIDUAL_CROPS = False     # Temporarily disable

# Important keys that we'll try to get from GPT if local OCR fails
LLM_IMPORTANT_KEYS = [
    "card1_rank", "card1_suit", "card2_rank", "card2_suit",
    "my_stack", "my_bet", "villian1_name", "villian1_stack", "villian1_bet",
    "villian2_name", "villian2_stack", "villian2_bet",
    "villian3_name", "villian3_stack", "villian3_bet",
    "villian4_name", "villian4_stack", "villian4_bet",
    "villian5_name", "villian5_stack", "villian5_bet",
    "total_pot", "game_id"
]

# keep classes mapping (kept from your original)
CLASSES = {
    i: name
    for i, name in enumerate([
        "button_allin", "button_bet", "button_call", "button_check", "button_fold",
        "button_raise", "card1_rank", "card1_suite_club", "card1_suite_diamond",
        "card1_suite_heart", "card1_suite_spades", "card2_rank", "card2_suite_club",
        "card2_suite_diamond", "card2_suite_heart", "card2_suite_spades",
        "flop1_rank", "flop1_suite_club", "flop1_suite_diamond", "flop1_suite_heart",
        "flop1_suite_spades", "flop2_rank", "flop2_suite_club", "flop2_suite_diamond",
        "flop2_suite_heart", "flop2_suite_spades", "flop3_rank", "flop3_suite_club",
        "flop3_suite_diamond", "flop3_suite_heart", "flop3_suite_spades", "game_id",
        "iinput_field", "my_bet", "my_stack", "position_BB", "position_SB",
        "river_rank", "river_suite_club", "river_suite_diamond", "river_suite_heart",
        "river_suite_spades", "total_pot", "turn_rank", "turn_suite_club",
        "turn_suite_diamond", "turn_suite_heart", "turn_suite_spades",
        "villian1_bet", "villian1_name", "villian1_stack", "villian2_bet",
        "villian2_name", "villian2_stack", "villian3_bet", "villian3_name",
        "villian3_stack", "villian4_bet", "villian4_name", "villian4_stack",
        "villian5_bet", "villian5_name", "villian5_stack", "winner"
    ])
}

# YOLO to JSON mapping (kept as you provided)
def _suit_keys(prefix: str):
    return [
        f"{prefix}_suit_club", f"{prefix}_suite_club",
        f"{prefix}_suit_diamond", f"{prefix}_suite_diamond",
        f"{prefix}_suit_heart", f"{prefix}_suite_heart",
        f"{prefix}_suit_spade", f"{prefix}_suite_spade",
        f"{prefix}_suit_spades", f"{prefix}_suite_spades",
    ]

def _rank_keys(prefix: str):
    return [f"{prefix}_rank", f"{prefix}_rank_area"]

YOLO_TO_JSON_MAP = {
    "button_fold": ("buttons", "Fold"),
    "button_check": ("buttons", "Check"),
    "button_bet": ("buttons", "Bet"),
    "button_raise": ("buttons", "Raise"),
    "button_call": ("buttons", "Call"),
    "button_allin": ("buttons", "All-in"),
}

for key in _rank_keys("card1"):
    YOLO_TO_JSON_MAP[key] = ("card1_rank", None)
for k in _suit_keys("card1"):
    if "club" in k:
        YOLO_TO_JSON_MAP[k] = ("card1_suit", "c")
    elif "diamond" in k:
        YOLO_TO_JSON_MAP[k] = ("card1_suit", "d")
    elif "heart" in k:
        YOLO_TO_JSON_MAP[k] = ("card1_suit", "h")
    elif "spade" in k:
        YOLO_TO_JSON_MAP[k] = ("card1_suit", "s")

for key in _rank_keys("card2"):
    YOLO_TO_JSON_MAP[key] = ("card2_rank", None)
for k in _suit_keys("card2"):
    if "club" in k:
        YOLO_TO_JSON_MAP[k] = ("card2_suit", "d")
    elif "diamond" in k:
        YOLO_TO_JSON_MAP[k] = ("card2_suit", "d")
    elif "heart" in k:
        YOLO_TO_JSON_MAP[k] = ("card2_suit", "h")
    elif "spade" in k:
        YOLO_TO_JSON_MAP[k] = ("card2_suit", "s")

for street in ["flop1", "flop2", "flop3", "turn", "river"]:
    for key in _rank_keys(street):
        YOLO_TO_JSON_MAP[key] = (f"{street}_rank", None)
    for k in _suit_keys(street):
        if "club" in k:
            YOLO_TO_JSON_MAP[k] = (f"{street}_suit", "c")
        elif "diamond" in k:
            YOLO_TO_JSON_MAP[k] = (f"{street}_suit", "d")
        elif "heart" in k:
            YOLO_TO_JSON_MAP[k] = (f"{street}_suit", "h")
        elif "spade" in k:
            YOLO_TO_JSON_MAP[k] = (f"{street}_suit", "s")


# ==================== OCR ENGINE (fixed __init__) ====================
class PokerOCR:
    def __init__(self):  # Fixed: _init_ -> __init__
        if USE_OCR:
            print("🔥 Initializing Enhanced OCR Engine...")
            try:
                import torch  # noqa: F401
                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False
            # easyocr reader init
            try:
                self.reader = easyocr.Reader(
                    ["en"],
                    gpu=use_gpu,
                    model_storage_directory="custom_ocr_models",
                    download_enabled=False
                )
            except Exception as e:
                print("⚠ easyocr init failed:", e)
                self.reader = None
        else:
            self.reader = None

        self.card_pattern = re.compile(r"^(A|K|Q|J|T|10|[2-9])([SHDCshdc♠♥♦♣])$", re.IGNORECASE)
        self.numeric_pattern = re.compile(r"[\d,.]+[kKmMbB]?")
        self.pot_pattern = re.compile(r"pot[:]?\s*([\d,.kKmMbM]+)", re.IGNORECASE)
        self.name_pattern = re.compile(r"^[a-zA-Z0-9_]{2,25}$")

    def process_detection(self, class_name, region):
        if not USE_OCR or self.reader is None:
            return None
        try:
            class_name_low = class_name.lower()
            if class_name_low.endswith("_rank") or class_name_low == "game_id":
                return self._extract_card_value(region)
            if (
                class_name_low.endswith("_bet")
                or class_name_low.endswith("_stack")
                or class_name_low in ("my_bet", "my_stack", "total_pot", "iinput_field")
            ):
                return self._extract_numeric_value(region)
            if class_name_low.endswith("_name") or ("villian" in class_name_low and "_name" in class_name_low):
                return self._extract_name(region)
        except Exception as e:
            print(f"⚠ OCR Error for {class_name}: {str(e)}")
        return None

    def _preprocess_region(self, region, is_card=False):
        if region is None or region.size == 0:
            return None
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrasted = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrasted, (3, 3), 0)
        if is_card:
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpen = cv2.filter2D(blurred, -1, sharpen_kernel)
            return sharpen
        return blurred

    def _binarize_options(self, img):
        out = []
        try:
            _, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ad = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            out.extend([th1, th2, ad])
            kernel = np.ones((2, 2), np.uint8)
            out.extend([cv2.morphologyEx(o, cv2.MORPH_OPEN, kernel) for o in out])
        except Exception:
            out = [img]
        return out

    def _extract_card_value(self, region):
        processed = self._preprocess_region(region, is_card=True)
        if processed is None:
            return None
        candidates = self._binarize_options(processed)
        for cand in candidates:
            for scale in [1.0, 1.5, 2.0]:
                try:
                    scaled = cv2.resize(cand, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                except Exception:
                    scaled = cand
                try:
                    res = self.reader.readtext(
                        scaled,
                        allowlist="AKQJT2345678910SHDCshdc♠♥♦♣",
                        detail=0,
                        paragraph=False
                    )
                except Exception:
                    res = []
                if res:
                    text = "".join(res).upper().replace("10", "T")
                    text = text.replace("♠", "S").replace("♥", "H").replace("♦", "D").replace("♣", "C")
                    text = re.sub(r"\s+", "", text.strip())
                    m = self.card_pattern.match(text)
                    if m:
                        rank = m.group(1).upper()
                        suit_char = m.group(2)[0].upper()
                        suit_map = {"S": "s", "H": "h", "D": "d", "C": "c"}
                        return rank + suit_map.get(suit_char, suit_char.lower())
                    r_match = re.search(r"(A|K|Q|J|T|10|[2-9])", text)
                    s_match = re.search(r"([SHDCshdc])", text)
                    if r_match and s_match:
                        rank = r_match.group(1).replace("10", "T").upper()
                        suit = s_match.group(1).upper()
                        suit_map = {"S": "s", "H": "h", "D": "d", "C": "c"}
                        return rank + suit_map.get(suit, suit.lower())
        # pytesseract fallback
        if TESSERACT_AVAILABLE:
            try:
                cfg = "--psm 7 -c tessedit_char_whitelist=AKQJT2345678910SHDCshdc"
                t = pytesseract.image_to_string(processed, config=cfg)
                t = re.sub(r"\s+", "", t.upper().replace("10", "T").strip())
                m = self.card_pattern.match(t)
                if m:
                    rank = m.group(1).upper()
                    suit = m.group(2)[0].upper()
                    suit_map = {"S": "s", "H": "h", "D": "d", "C": "c"}
                    return rank + suit_map.get(suit, suit.lower())
            except Exception:
                pass
        return None

    def _extract_numeric_value(self, region):
        processed = self._preprocess_region(region, is_card=False)
        if processed is None:
            return None
        for cand in self._binarize_options(processed):
            try:
                res = self.reader.readtext(cand, allowlist="0123456789.,kKmMbB$", detail=0, paragraph=False)
            except Exception:
                res = []
            if res:
                text = "".join(res).upper().replace("$", "").replace("O", "0").replace("I", "1").strip()
                text = text.replace(",", "")
                match = re.search(r"[\d.]+[kKmMbB]?", text)
                if match:
                    return match.group()
        if TESSERACT_AVAILABLE:
            try:
                cfg = "--psm 7 -c tessedit_char_whitelist=0123456789.,KkMm"
                t = pytesseract.image_to_string(processed, config=cfg)
                t = t.upper().replace(",", "").strip()
                m = re.search(r"[\d.]+[kKmMbB]?", t)
                if m:
                    return m.group()
            except Exception:
                pass
        return None

    def _extract_name(self, region):
        processed = self._preprocess_region(region)
        if processed is None:
            return None
        try:
            res = self.reader.readtext(
                processed,
                allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
                detail=0,
                paragraph=False
            )
        except Exception:
            res = []
        if res:
            text = re.sub(r"[^A-Za-z0-9_]", "", res[0].strip())
            if self.name_pattern.match(text):
                return text
        return None

# ==================== SCREENSHOT CAPTURE ====================


# ==================== HELPERS ====================

def clean_rank(rank: str) -> str:
    if not rank:
        return ""
    rank = rank.strip().upper()
    corrections = {"0": "Q", "X": "K", "1": "I", "O": "Q"}
    return corrections.get(rank, rank)


def write_json_atomic(path: str, data: dict):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_json_", dir=(dirpath if dirpath else None))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ==================== POKER DETECTOR ====================


class PokerDetector:
    def __init__(self, model_path, classes, confidence_threshold, ocr_engine):
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf_threshold = confidence_threshold
        self.ocr = ocr_engine

    def process_screenshot(self, image_path, output_json, output_image):
        frame = cv2.imread(image_path)
        results = self.model(frame, conf=self.conf_threshold)

        card_ranks = {}
        card_suits = {}
        community_cards = {}
        buttons = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            class_name = self.classes[class_id]
            confidence = float(box.conf)
            region = frame[y1:y2, x1:x2]

            ocr_text = None
            if class_name in [
                'card1_rank', 'card2_rank', 'flop1_rank', 'flop2_rank', 'flop3_rank',
                'turn_rank', 'river_rank', 'total_pot', 'my_bet', 'my_stack',
                'villian1_bet', 'villian2_bet', 'villian3_bet', 'villian4_bet', 'villian5_bet',
                'villian1_name', 'villian2_name', 'villian3_name', 'villian4_name', 'villian5_name',
                'villian1_stack', 'villian2_stack', 'villian3_stack', 'villian4_stack', 'villian5_stack',
                'game_id'
            ]:
                ocr_text = self.ocr.process_detection(class_name, region)

            # Store ranks and suits separately
            if "_rank" in class_name and ocr_text:
                card_ranks[class_name] = ocr_text
            elif "_suite_" in class_name:
                suit = class_name.split("_suite_")[-1][0]
                card_suits[class_name.replace("_suite_", "_rank")] = suit

            # Community cards
            if class_name.startswith(('flop', 'turn', 'river')):
                if "_rank" in class_name and ocr_text:
                    community_cards[class_name] = ocr_text + card_suits.get(class_name, '')

            # Buttons
            if class_name.startswith("button_"):
                cx, cy = (x1+x2)//2, (y1+y2)//2
                buttons.append({"button": class_name, "center": [cx, cy]})

            # Draw annotations
            label = f"{class_name}:{ocr_text if ocr_text else ''}"
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        # Merge hole cards
        card1 = card_ranks.get("card1_rank", "") + card_suits.get("card1_rank", "")
        card2 = card_ranks.get("card2_rank", "") + card_suits.get("card2_rank", "")

        # Determine game state
        comm_count = len([c for c in community_cards.values() if c])
        if comm_count == 0:
            game_state = "PREFLOP"
        elif comm_count == 3:
            game_state = "FLOP"
        elif comm_count == 4:
            game_state = "TURN"
        else:
            game_state = "RIVER"

        # Build final JSON
        result = {
            "game_id": card_ranks.get("game_id", ""),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "my_stack": card_ranks.get("my_stack", ""),
            "card1": card1,
            "card2": card2,
            "my_bet": card_ranks.get("my_bet", ""),
            "villains": [],
            "buttons": buttons,
            "community_cards": list(community_cards.values()),
            "game_state": game_state
        }

        # Villains 1–5
        for i in range(1, 6):
            villain = {
                "name": card_ranks.get(f"villian{i}_name", ""),
                "stack": card_ranks.get(f"villian{i}_stack", ""),
                "bet": card_ranks.get(f"villian{i}_bet", "")
            }
            result["villains"].append(villain)

        # Save outputs
        with open(output_json, "w") as f:
            json.dump(result, f, indent=4)

        cv2.imwrite(output_image, frame)
        print(f"✅ JSON saved to {output_json}")
        print(f"✅ Annotated screenshot saved to {output_image}")

        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            x = c * w_max
            y = r * (h_max + 18)
            collage.paste(img, (x, y))
            # draw label below
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(collage)
                font = ImageFont.load_default()
                draw.text((x + 2, y + img.height + 2), labels[idx], font=font, fill=(255, 255, 255))
            except Exception:
                pass

        tmpfd, tmp_path = tempfile.mkstemp(prefix="poker_collage_", suffix=".jpg", dir=".")
        os.close(tmpfd)
        collage.save(tmp_path, quality=85)
        if DEBUG:
            print("Collage saved to", tmp_path)
        return tmp_path

    def merge_detected_values(self, existing: dict, new: dict):
        merged = dict(existing) if existing else {}
        for k in set(list(new.keys()) + list(merged.keys())):
            merged.setdefault(k, "")

        changes = {"cards_filled": [], "other_updated": []}

        if isinstance(new.get("buttons"), list) and len(new["buttons"]) > 0:
            merged["buttons"] = new["buttons"]

        for k, v in new.items():
            if k == "buttons":
                continue
            if k in self.card_keys:
                if not merged.get(k) and v:
                    merged[k] = v
                    changes["cards_filled"].append(k)
            else:
                if isinstance(v, str):
                    if v.strip() and merged.get(k) != v:
                        merged[k] = v
                        changes["other_updated"].append(k)
                else:
                    if (v or v == 0) and merged.get(k) != v:
                        merged[k] = v
                        changes["other_updated"].append(k)

        return merged, changes

    def build_result(self, detected_values: dict, raw_detections: list) -> dict:
        # Build clean card representations
        card1 = (detected_values.get("card1_rank", "") + detected_values.get("card1_suit", "")).strip()
        card2 = (detected_values.get("card2_rank", "") + detected_values.get("card2_suit", "")).strip()

        # Build community cards in order
        community_cards = []
        for prefix in ["flop1", "flop2", "flop3", "turn", "river"]:
            rank = detected_values.get(f"{prefix}_rank", "")
            suit = detected_values.get(f"{prefix}_suit", "")
            if rank:
                community_cards.append((rank + suit).strip())

        # Determine game state
        if detected_values.get("river_rank"):
            game_state = "RIVER"
        elif detected_values.get("turn_rank"):
            game_state = "TURN"
        elif any(detected_values.get(k) for k in ["flop1_rank", "flop2_rank", "flop3_rank"]):
            game_state = "FLOP"
        else:
            game_state = "PREFLOP"

        # Build villains array in order
        villains = []
        for vi in range(1, 6):
            villains.append({
                "name": detected_values.get(f"villian{vi}_name", ""),
                "stack": detected_values.get(f"villian{vi}_stack", ""),
                "bet": detected_values.get(f"villian{vi}_bet", "")
            })

        # Create clean, simple JSON output
        result = {
            "game_info": {
                "game_id": detected_values.get("game_id", ""),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "game_state": game_state
            },
            "my_cards": {
                "card1": card1,
                "card2": card2
            },
            "my_info": {
                "stack": detected_values.get("my_stack", ""),
                "bet": detected_values.get("my_bet", "")
            },
            "villains": villains,
            "community_cards": community_cards,
            "buttons": detected_values.get("buttons", []),
            "pot": detected_values.get("total_pot", "")
        }

        return result

    def run_live(self, output_json="poker_result.json", output_image="poker_labeled.png"):
        if self.model is None:
            print("❌ Model is not loaded. Exiting.")
            return

        output_json_path = os.path.abspath(output_json)
        output_image_path = os.path.abspath(output_image)
        mode_desc = "FULL SCREEN" if FULLSCREEN else f"REGION {SCREEN_REGION}"
        print(f"🎬 Starting live capture on {mode_desc}.")
        print(f"📄 JSON path: {output_json_path}")
        print(f"🖼  Image path: {output_image_path}")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                try:
                    if FULLSCREEN:
                        pil_img = pyautogui.screenshot()
                    else:
                        pil_img = pyautogui.screenshot(region=SCREEN_REGION)
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"❌ Screenshot failed: {e}")
                    time.sleep(LOOP_INTERVAL_SECONDS)
                    continue

                new_detected_values, raw_detections, annotated, stats = self.process_frame(frame)
                
                # Read existing result
                existing_result = None
                if os.path.exists(output_json_path):
                    try:
                        with open(output_json_path, "r", encoding="utf-8") as f:
                            existing_result = json.load(f)
                    except Exception as e:
                        print(f"⚠ Failed to read existing JSON: {e}")
                
                # Accumulate data instead of replacing
                if ACCUMULATE_DATA:
                    existing_detected = (existing_result or {}).get("detected_values", {})
                    merged_detected = self.accumulate_detected_data(existing_detected, new_detected_values)
                else:
                    merged_detected = new_detected_values
                
                result = self.build_result(merged_detected, raw_detections)

                try:
                    write_json_atomic(output_json_path, result)
                    ok = cv2.imwrite(output_image_path, annotated)
                    size = os.path.getsize(output_json_path) if os.path.exists(output_json_path) else 0
                    print(
                        f"✅ Updated {output_json_path} at {result['game_info']['time']} "
                        f"(boxes={stats['boxes']}, non_empty_fields={stats['non_empty_fields']}, "
                        f"cards_filled={len(changes['cards_filled'])}, other_updated={len(changes['other_updated'])}, "
                        f"bytes={size}) — next in {LOOP_INTERVAL_SECONDS}s"
                    )
                    if not ok:
                        print("⚠ Failed to save annotated image.")
                except Exception as e:
                    print(f"❌ Failed to save outputs: {e}")

                time.sleep(LOOP_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n🛑 Stopped live capture.")


# ==================== GPT-4o Integration ====================
def query_gpt4o_for_crops(collage_path: str, missing_keys: List[str]) -> dict:
    """
    Send the collage to GPT-4o for better text detection and return JSON mapping.
    Updated for OpenAI Python library 1.0.0+
    """
    if not OPENAI_AVAILABLE:
        print("⚠ OpenAI SDK not installed; skipping GPT-4o fallback.")
        return {}

    if not os.path.exists(collage_path):
        print("⚠ Collage missing; skipping GPT-4o fallback.")
        return {}

    # Check for API key from .env file
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY not found in .env file; skipping GPT-4o fallback.")
        print(" Make sure you have a .env file with OPENAI_API_KEY=your_key_here")
        return {}

    try:
        # Create OpenAI client with new API
        client = openai.OpenAI(api_key=api_key)
        print(f"🔑 Using OpenAI API key: {api_key[:10]}...")

        # Create a comprehensive prompt for GPT-4o
        system_prompt = """You are an expert data extraction specialist. 
        Analyze collage screenshots and extract precise information.
        Return ONLY valid JSON with no additional text."""

        user_prompt = f"""Analyze this image collage carefully. 

        EXTRACT THESE FIELDS IF VISIBLE: {', '.join(missing_keys)}

        CRITICAL FORMATTING RULES:
        - Card ranks: Use A, K, Q, J, T (for 10), 2-9
        - Card suits: Use c (clubs), d (diamonds), h (hearts), s (spades)
        - Complete cards: Combine rank + suit like "As" (Ace of spades), "Th" (Ten of hearts)
        - Numeric values: Return exactly as shown (e.g., "1.2k", "1500", "$500", "2.5M")
        - Player names: Return exactly as shown (e.g., "Player123", "John_Doe")
        - Game IDs: Return exactly as shown (e.g., "12345", "Game_ABC")
        - If a field cannot be read clearly, do NOT include it in the JSON
        - Return ONLY valid JSON, no other text, quotes, or formatting

        EXPECTED OUTPUT FORMAT:
        {{
            "card1_rank": "A",
            "card1_suit": "s",
            "my_stack": "1500",
            "villian1_name": "PlayerX",
            "total_pot": "450"
        }}

        ANALYSIS INSTRUCTIONS:
        1. Look at each cropped region carefully
        2. Identify what type of information each region contains
        3. Extract text values with maximum accuracy
        4. For cards, separate rank and suit when possible
        5. For numbers, preserve exact formatting (including $, k, M, etc.)
        6. For names, preserve exact spelling and case
        7. Only include fields that are clearly readable

        IMPORTANT: This is a collage  screenshot. Look for:
        - Playing cards (ranks 2-10, J, Q, K, A and suits ♠♥♦♣)
        - Player information (names, stack amounts, bet amounts)
        - Game information (pot size, game ID, position indicators)
        - Action buttons (Fold, Call, Raise, Check, All-in)

        Return ONLY the JSON object with no additional text, explanations, or formatting."""

        # Read the image file
        with open(collage_path, "rb") as image_file:
            # Use the new OpenAI API format for GPT-4o
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"}}
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=1500
            )

        # Extract the response text (new API structure)
        response_text = response.choices[0].message.content.strip()
        print(f"✅ GPT-4o response: {response_text[:100]}...")
        
        # Parse JSON response
        try:
            # First try direct JSON parsing
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                print(f"✅ GPT-4o successfully extracted data: {list(parsed.keys())}")
                return parsed
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, dict):
                        print(f"✅ GPT-4o extracted data from response: {list(parsed.keys())}")
                        return parsed
                except json.JSONDecodeError:
                    pass
        
        print(f"⚠ Could not parse GPT-4o response as JSON. Response: {response_text[:200]}...")
        return {}

    except Exception as e:
        print(f"⚠ Exception during GPT-4o call: {e}")
        print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        return {}


# ==================== MAIN ====================
if __name__ == "__main__":
    print(" Starting Poker OCR + Detector with GPT-4o integration")
    print("📁 Loading environment variables from .env file...")
    
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
    detector = PokerDetector("poker_model.pt", CLASSES, 0.5, ocr_engine=PokerOCR())
    detector.process_screenshot(
        image_path="test_screenshot.png",
        output_json="poker_result.json",
        output_image="poker_labeled.png"
    )
