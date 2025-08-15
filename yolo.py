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

# Optional: for pytesseract fallback
try:
    import pytesseract  # noqa: F401
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# OpenAI integration - Make sure to set OPENAI_API_KEY in environment
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

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


# ==================== DETECTOR (fixed __init__) ====================
class PokerDetector:
    def __init__(self, model_path, classes, confidence_threshold, ocr_engine: PokerOCR = None):  # Fixed: _init_ -> __init__
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"⚠ Warning: Could not load model {model_path}: {e}")
            self.model = None
        self.classes = classes
        self.conf_threshold = confidence_threshold
        self.ocr = ocr_engine or PokerOCR()
        self.card_keys = [
            "card1_rank", "card1_suit", "card2_rank", "card2_suit",
            "flop1_rank", "flop1_suit", "flop2_rank", "flop2_suit",
            "flop3_rank", "flop3_suit", "turn_rank", "turn_suit",
            "river_rank", "river_suit",
        ]

    # small helper: crop with pad and bounds-check
    @staticmethod
    def _safe_crop(frame, x1, y1, x2, y2, pad=2):
        h, w = frame.shape[:2]
        x1c = max(0, int(x1) - pad)
        y1c = max(0, int(y1) - pad)
        x2c = min(w, int(x2) + pad)
        y2c = min(h, int(y2) + pad)
        if x2c <= x1c or y2c <= y1c:
            return None
        return frame[y1c:y2c, x1c:x2c]

    def run_ocr_on_bbox(self, bbox_img):
        if self.ocr is None or self.ocr.reader is None:
            return ""
        gray = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        try:
            text = self.ocr.reader.readtext(thresh, detail=0)
        except Exception:
            text = []
        return clean_rank(text[0]) if text else ""

    def _iter_detections(self, results):
        if results is None:
            return
        try:
            res = results[0]
        except Exception:
            res = results
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return
        try:
            n = len(boxes)
        except Exception:
            n = 0
        for i in range(n):
            b = boxes[i]
            try:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
            except Exception:
                try:
                    x1, y1, x2, y2 = map(int, b.xyxy.view(-1).tolist()[:4])
                except Exception:
                    continue
            try:
                c = b.conf
                conf = float(c[0].item() if hasattr(c, "len") else c.item())
            except Exception:
                try:
                    conf = float(b.conf)
                except Exception:
                    conf = None
            try:
                cls_t = b.cls
                class_id = int(cls_t[0].item() if hasattr(cls_t, "len") else cls_t.item())
            except Exception:
                try:
                    class_id = int(float(b.cls))
                except Exception:
                    class_id = -1
            yield {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": class_id, "conf": conf}

    def process_frame(self, frame):
        """
        Enhanced frame processing with individual card cropping and collage analysis.
        """
        if self.model is None:
            print("⚠ Model not loaded, skipping detection stage.")
            return {}, [], frame, {"boxes": 0, "non_empty_fields": 0}

        try:
            results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        except Exception:
            results = self.model(frame, conf=self.conf_threshold)

        new_detected_values = {}
        for k, v in YOLO_TO_JSON_MAP.items():
            key, _ = v
            if key != "buttons":
                new_detected_values.setdefault(key, "")
        for fld in [
            "game_id", "my_bet", "my_stack", "total_pot",
            "villian1_name", "villian1_stack", "villian1_bet",
            "villian2_name", "villian2_stack", "villian2_bet",
            "villian3_name", "villian3_stack", "villian3_bet",
            "villian4_name", "villian4_stack", "villian4_bet",
            "villian5_name", "villian5_stack", "villian5_bet",
        ]:
            new_detected_values.setdefault(fld, "")
        new_detected_values["buttons"] = []

        raw_detections = []
        box_count = 0

        # collect crops for collage/GPT fallback
        crops_for_fallback = []  # list of (class_name, bbox, crop_img)

        for i, det in enumerate(self._iter_detections(results)):
            box_count += 1
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            class_id = det["class_id"]
            conf = det["conf"]
            class_name = self.classes.get(class_id, f"class{class_id}")

            y1c, y2c = max(0, y1), max(0, y2)
            x1c, x2c = max(0, x1), max(0, x2)
            region = frame[y1c:y2c, x1c:x2c]

            if DEBUG:
                os.makedirs("debug_crops", exist_ok=True)
                cv2.imwrite(f"debug_crops/{i}_{class_name}.png", region if region is not None else np.zeros((2,2,3), np.uint8))

            mapped_to = None
            mapped_value = None
            ocr_text = None

            if class_name in YOLO_TO_JSON_MAP:
                key, val = YOLO_TO_JSON_MAP[class_name]
                mapped_to = key
                if key == "buttons":
                    new_detected_values["buttons"].append(val)
                    mapped_value = val
                elif val is not None:
                    new_detected_values[key] = val
                    mapped_value = val
                else:
                    # run OCR locally on bbox
                    rank_val = None
                    if region is not None and region.size != 0:
                        rank_val = self.run_ocr_on_bbox(region)
                    new_detected_values[key] = rank_val or ""
                    mapped_value = rank_val
                    ocr_text = rank_val
            else:
                # If YOLO did not map to known key, let OCR engine try to extract.
                ocr_text = self.ocr.process_detection(class_name, region) if self.ocr else None
                if ocr_text:
                    new_detected_values[class_name] = ocr_text
                if "suite" in class_name.lower() or "suit" in class_name.lower():
                    suit_word = class_name.split("_")[-1].lower()
                    suit_map = {"club": "c", "diamond": "d", "heart": "h", "spade": "s", "spades": "s"}
                    for tgt_prefix in ["card1", "card2", "flop1", "flop2", "flop3", "turn", "river"]:
                        if class_name.startswith(tgt_prefix):
                            new_detected_values[f"{tgt_prefix}_suit"] = suit_map.get(suit_word, "?")
                            mapped_to = f"{tgt_prefix}_suit"
                            mapped_value = new_detected_values[f"{tgt_prefix}_suit"]
                            break

            # Keep crops for LLM fallback if the channel is likely important
            if any(k in class_name.lower() for k in ("name", "stack", "bet", "rank", "suite", "suit")):
                crop_img = self._safe_crop(frame, x1, y1, x2, y2)
                if crop_img is not None:
                    crops_for_fallback.append((class_name, (x1, y1, x2, y2), crop_img))

            # Annotate
            label_bits = [class_name]
            if ocr_text:
                label_bits.append(str(ocr_text))
            if mapped_to and mapped_value and mapped_to != "buttons":
                label_bits.append(f"{mapped_to}:{mapped_value}")
            label = " ".join(label_bits)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            raw_detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "ocr_text": ocr_text,
                "mapped_to": mapped_to,
                "mapped_value": mapped_value,
            })

        non_empty_fields = sum(
            1 for k, v in new_detected_values.items()
            if k != "buttons" and isinstance(v, str) and v.strip()
        )
        stats = {"boxes": box_count, "non_empty_fields": non_empty_fields}

        # If important keys are empty and we have crops, create collage and call GPT-4o
        missing_keys = [k for k in LLM_IMPORTANT_KEYS if not new_detected_values.get(k)]
        if USE_GPT_FALLBACK and crops_for_fallback and missing_keys:
            try:
                collage_path = self._build_collage(crops_for_fallback)
                # Call GPT-4o for better accuracy
                llm_results = query_gpt4o_for_crops(collage_path, missing_keys)
                if isinstance(llm_results, dict):
                    for k, v in llm_results.items():
                        if v and not new_detected_values.get(k):
                            new_detected_values[k] = v
                            print(f"✅ GPT-4o filled missing field: {k} = {v}")
            except Exception as e:
                print("⚠ GPT-4o fallback failed:", e)

        # After collecting detections, crop all regions
        # Comment out this section temporarily:
        # if CROP_INDIVIDUAL_CARDS or CREATE_DETECTION_COLLAGE:
        #     individual_crops, community_crops, other_elements = self.crop_all_detected_regions(frame, raw_detections)
            
        #     # Analyze individual hole cards
        #     if individual_crops:
        #         gpt_card_results = self.analyze_individual_cards_with_gpt4o(individual_crops)
                
        #         # Update detected values with individual card results
        #         for card_type, card_data in gpt_card_results.items():
        #             if "hole_card_1" in card_type:
        #                 new_detected_values["card1_rank"] = card_data["rank"]
        #                 new_detected_values["card1_suit"] = card_data["suit"]
        #             elif "hole_card_2" in card_type:
        #                 new_detected_values["card2_rank"] = card_data["rank"]
        #                 new_detected_values["card2_suit"] = card_data["suit"]
            
        #     # Analyze community cards and other elements with collage
        #     if community_crops or other_elements:
        #         collage_results = self.analyze_collage_with_gpt4o(community_crops, other_elements)
                
        #         # Update with collage results
        #         if collage_results:
        #             # Update community cards
        #             if "community_cards" in collage_results:
        #                 for card_name, card_data in collage_results["community_cards"].items():
        #                     if "flop1" in card_name:
        #                         new_detected_values["flop1_rank"] = card_data.get("rank", "")
        #                         new_detected_values["flop1_suit"] = card_data.get("suit", "")
        #                     elif "flop2" in card_name:
        #                         new_detected_values["flop2_rank"] = card_data.get("rank", "")
        #                         new_detected_values["flop2_suit"] = card_data.get("suit", "")
        #                     elif "flop3" in card_name:
        #                         new_detected_values["flop3_rank"] = card_data.get("rank", "")
        #                         new_detected_values["flop3_suit"] = card_data.get("suit", "")
        #                     elif "turn" in card_name:
        #                         new_detected_values["turn_rank"] = card_data.get("rank", "")
        #                         new_detected_values["turn_suit"] = card_data.get("suit", "")
        #                     elif "river" in card_name:
        #                         new_detected_values["river_rank"] = card_data.get("rank", "")
        #                         new_detected_values["river_suit"] = card_data.get("suit", "")
                    
        #             # Update other information
        #             if "player_info" in collage_results:
        #                 for key, value in collage_results["player_info"].items():
        #                     if key == "my_stack":
        #                         new_detected_values["my_stack"] = value
        #                     elif key == "my_bet":
        #                         new_detected_values["my_bet"] = value
        #                     elif key == "pot":
        #                         new_detected_values["total_pot"] = value
                    
        #             if "buttons" in collage_results:
        #                 new_detected_values["buttons"] = collage_results["buttons"]

        return new_detected_values, raw_detections, frame, stats

    def _build_collage(self, crops: List[Tuple[str, Tuple[int,int,int,int], np.ndarray]], thumb_size=(160, 60)):
        """
        Create a tiled collage of crops and save to a temp file. Return path.
        crops: list of (class_name, bbox, img)
        """
        images = []
        labels = []
        for class_name, bbox, img in crops:
            try:
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pil.thumbnail(thumb_size)
                images.append(pil)
                labels.append(class_name)
            except Exception:
                continue

        if not images:
            raise ValueError("No valid crops to build collage.")

        # compute grid
        n = len(images)
        cols = min(4, n)
        rows = ceil(n / cols)
        w_max = max(img.width for img in images)
        h_max = max(img.height for img in images)

        collage_w = cols * w_max
        collage_h = rows * (h_max + 18)  # extra for label text
        collage = Image.new("RGB", (collage_w, collage_h), (40, 40, 40))

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
    
    # Check if .env file exists
    if os.path.exists(".env"):
        print("✅ .env file found")
    else:
        print("⚠ .env file not found. Create one with OPENAI_API_KEY=your_key_here")
    
    detector = PokerDetector(MODEL_PATH, CLASSES, CONFIDENCE_THRESHOLD, ocr_engine=PokerOCR())
    detector.run_live(output_json="poker_result.json", output_image="poker_labeled.png")