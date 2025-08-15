# poker_live_pipeline.py
# Live YOLO + SAHI-on-demand + DeepSORT + OCR pipeline for poker screenshots
# Outputs one JSON per hand containing only buttons coords + iinput_field coords.

import time
import os
import json
from collections import defaultdict, deque, Counter
from typing import List, Dict, Tuple

import cv2
import numpy as np
import mss
import easyocr

from ultralytics import YOLO

# DeepSORT realtime - may need pip install deep-sort-realtime
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except Exception:
    DEEPSORT_AVAILABLE = False

# SAHI for sliced inference (optional)
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except Exception:
    SAHI_AVAILABLE = False

# ----------------- CONFIG -----------------
MODEL_PATH = "poker_model.pt"           # path to your YOLO model
SCREEN_REGION = {"left": 0, "top": 29, "width": 770, "height": 543}
INPUT_FPS = 6                           # target processing frames per second
OUTPUT_FOLDER = "hand_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# YOLO / SAHI
YOLO_IMGSZ = 1280
YOLO_CONF = 0.35
SAHI_SLICE_H = 640
SAHI_SLICE_W = 640
SAHI_OVERLAP_RATIO = 0.2               # 20% overlap (you asked for 20–30%)

# DeepSORT params
DEEPSORT_MAX_AGE = 6
DEEPSORT_N_INIT = 1
DEEPSORT_MAX_COSINE_DIST = 0.25        # tune 0.2-0.4; lower => stricter appearance match
DEEPSORT_NN_BUDGET = 100

# OCR
USE_OCR = True
OCR_LANGS = ["en"]
OCR_GPU = False                         # set True if you have GPU and EasyOCR compiled with GPU support
ocr_reader = easyocr.Reader(OCR_LANGS, gpu=OCR_GPU)

# Hand/session logic
HAND_TIMEOUT = 6.0  # seconds of no button detection -> finalize a hand
GAME_ID_CLASS_NAME = "game_id"  # YOLO label name if present

# Classes of interest (prefix / exact names must match your training labels)
BUTTON_CLASS_PREFIX = "button_"
INPUT_FIELD_CLASS = "iinput_field"   # exact label from your dataset
SMALL_OBJ_HINT_CLASSES = ["my_bet", "total_pot", "villian1_bet"]  # if these are present we assume small text important

# Debug window flags
SHOW_DEBUG_WINDOW = True
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ----------------- Helpers -----------------

def preprocess_for_ocr(crop: np.ndarray, upscale: bool = True) -> np.ndarray:
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    blur = cv2.GaussianBlur(cl, (3, 3), 0)
    # deskew
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(th)
    if coords is not None:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = blur.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        blur = cv2.warpAffine(blur, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    if upscale:
        th2 = cv2.resize(th2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return th2

def parse_ultralytics_results(results, image_shape) -> List[Dict]:
    dets = []
    if results is None:
        return dets
    try:
        res = results[0]
    except Exception:
        res = results
    boxes = getattr(res, "boxes", None)
    names = getattr(res, "names", {})
    if boxes is None:
        return dets
    for b in boxes:
        try:
            xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else b.xyxy[0].numpy()
            x1, y1, x2, y2 = map(int, xyxy.tolist())
        except Exception:
            try:
                coords = b.xyxy.view(-1).tolist()
                x1, y1, x2, y2 = map(int, coords[:4])
            except Exception:
                continue
        try:
            conf = float(b.conf[0]) if hasattr(b.conf, "_len_") else float(b.conf)
        except Exception:
            conf = float(b.conf) if hasattr(b, "conf") else 0.0
        try:
            cls_id = int(b.cls[0]) if hasattr(b.cls, "_len_") else int(b.cls)
        except Exception:
            cls_id = int(float(b.cls)) if hasattr(b, "cls") else -1
        cls_name = names.get(cls_id, f"class{cls_id}")
        dets.append({
            "x1": max(0, x1), "y1": max(0, y1), "x2": min(image_shape[1]-1, x2), "y2": min(image_shape[0]-1, y2),
            "conf": conf, "class_id": cls_id, "class_name": cls_name
        })
    return dets

def avg_bbox(history_bboxes: deque) -> Tuple[int,int,int,int]:
    if not history_bboxes:
        return (0,0,0,0)
    xs1, ys1, xs2, ys2 = zip(*history_bboxes)
    return (int(sum(xs1)/len(xs1)), int(sum(ys1)/len(ys1)), int(sum(xs2)/len(xs2)), int(sum(ys2)/len(ys2)))

# ----------------- Pipeline -----------------

class LivePokerPipeline:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.sahi_model = None
        if SAHI_AVAILABLE:
            try:
                device = "cuda:0" if self.model.model.device.type == "cuda" else "cpu"
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8", model_path=model_path, device=device
                )
            except Exception as e:
                print("[SAHI] init failed:", e)
                self.sahi_model = None

        # tracker
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT,
                                    max_cosine_distance=DEEPSORT_MAX_COSINE_DIST, nn_budget=DEEPSORT_NN_BUDGET)
        else:
            self.tracker = None
            print("[WARN] deep-sort-realtime not available. Install it for better tracking.")

        # per-track history
        self.track_history = defaultdict(lambda: {"class_votes": deque(maxlen=7),
                                                  "bboxes": deque(maxlen=7),
                                                  "last_seen_ts": 0})

        # hand/session
        self.hand_index = 0
        self.hand_start_ts = None
        self.last_button_seen_ts = None
        self.last_game_id = None

    def run_yolo(self, frame: np.ndarray):
        results = self.model.predict(source=frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        dets = parse_ultralytics_results(results, frame.shape)
        return dets

    def run_sahi(self, frame: np.ndarray):
        if not SAHI_AVAILABLE or self.sahi_model is None:
            return None
        pred = get_sliced_prediction(frame, detection_model=self.sahi_model,
                                     slice_height=SAHI_SLICE_H, slice_width=SAHI_SLICE_W,
                                     overlap_height_ratio=SAHI_OVERLAP_RATIO, overlap_width_ratio=SAHI_OVERLAP_RATIO)
        out = []
        for obj in pred.object_prediction_list:
            x1 = int(obj.bbox.x_min); y1 = int(obj.bbox.y_min); x2 = int(obj.bbox.x_max); y2 = int(obj.bbox.y_max)
            conf = float(obj.confidence); cls_name = obj.category.name
            out.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf,"class_name":cls_name})
        return out

    def update_tracker(self, frame: np.ndarray, detections: List[Dict], ts: float):
        ds_input = []
        for d in detections:
            bbox = [d["x1"], d["y1"], d["x2"], d["y2"]]
            conf = d.get("conf", 0.0)
            cls = d.get("class_name", "unknown")
            ds_input.append((bbox, conf, cls))
        tracks = []
        if self.tracker is not None:
            tracks = self.tracker.update_tracks(ds_input, frame=frame)
        else:
            # fallback: simple pseudo-tracker - treat each detection as a separate track (not ideal)
            for i, d in enumerate(detections):
                class_name = d.get("class_name", "unknown")
                tracks.append(type("T", (), {"is_confirmed": lambda self=True: True,
                                             "track_id": i,
                                             "to_ltrb": lambda self=True, d=d: [d["x1"], d["y1"], d["x2"], d["y2"]],
                                             "det_class": class_name}))
        active_tracks = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            ltrb = tr.to_ltrb()
            cls = getattr(tr, "det_class", "unknown")
            h = self.track_history[tid]
            h["class_votes"].append(cls)
            h["bboxes"].append((int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])))
            h["last_seen_ts"] = ts
            active_tracks.append({"track_id": tid, "class": cls, "bbox": h["bboxes"][-1]})
        # cleanup stale histories
        stale = [tid for tid,h in self.track_history.items() if ts - h["last_seen_ts"] > 30]
        for s in stale:
            del self.track_history[s]
        return active_tracks

    def detect_buttons_input(self, active_tracks: List[Dict]):
        buttons = []
        input_area = None
        for t in active_tracks:
            tid = t["track_id"]
            votes = list(self.track_history[tid]["class_votes"])
            cls_label = Counter(votes).most_common(1)[0][0] if votes else t["class"]
            avg = avg_bbox(self.track_history[tid]["bboxes"])
            if cls_label.startswith(BUTTON_CLASS_PREFIX):
                buttons.append({"track_id": tid, "class": cls_label, "bbox": {"x1":avg[0],"y1":avg[1],"x2":avg[2],"y2":avg[3]}})
                self.last_button_seen_ts = time.time()
                if self.hand_start_ts is None:
                    self.hand_start_ts = time.time()
            if cls_label == INPUT_FIELD_CLASS:
                input_area = {"track_id": tid, "class": cls_label, "bbox": {"x1":avg[0],"y1":avg[1],"x2":avg[2],"y2":avg[3]}}
        return buttons, input_area

    def ocr_crop(self, frame: np.ndarray, bbox: Dict) -> str:
        x1,y1,x2,y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        crop = frame[y1:y2, x1:x2]
        proc = preprocess_for_ocr(crop, upscale=True)
        if proc is None:
            return ""
        try:
            res = ocr_reader.readtext(proc, detail=0, paragraph=False)
            if res:
                return " ".join(res)
        except Exception:
            return ""
        return ""

    def finalize_hand(self, buttons: List[Dict], input_area: Dict):
        self.hand_index += 1
        out = {"hand_index": self.hand_index,
               "time_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.hand_start_ts)) if self.hand_start_ts else None,
               "time_end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
               "buttons": [], "iinput_field": None}
        for b in buttons:
            out["buttons"].append({"track_id": int(b["track_id"]), "class": b["class"], "bbox": b["bbox"]})
        if input_area:
            out["iinput_field"] = {"track_id": int(input_area["track_id"]), "class": input_area["class"], "bbox": input_area["bbox"]}
        fname = os.path.join(OUTPUT_FOLDER, f"hand_{self.hand_index}_{int(time.time())}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("[HAND SAVED]", fname)
        # reset hand markers but keep tracker so continuity remains
        self.hand_start_ts = None
        self.last_button_seen_ts = None
        return fname

    def step(self, frame: np.ndarray):
        ts = time.time()
        # 1) quick YOLO
        dets = self.run_yolo(frame)
        # if YOLO returned few detections or we suspect small objects, run SAHI to recover tiny ones
        need_sahi = False
        if self.sahi_model is not None:
            if len(dets) < 6:  # heuristic: few detections -> maybe we missed small ones
                need_sahi = True
            else:
                # if any class in results is small-hint classes, run SAHI too
                if any(d["class_name"] in SMALL_OBJ_HINT_CLASSES for d in dets):
                    need_sahi = True
        if need_sahi and self.sahi_model is not None:
            try:
                sahi_dets = self.run_sahi(frame)
                if sahi_dets:
                    # merge: prefer SAHI detection list (they are more sensitive)
                    dets = sahi_dets
            except Exception as e:
                print("[SAHI] failed:", e)
        # 2) track update
        active = self.update_tracker(frame, dets, ts)
        # 3) detect buttons & input area (with averaged bbox)
        buttons, input_field = self.detect_buttons_input(active)
        # 4) hand session logic
        # attempt to read game_id if present
        game_ids = []
        for d in dets:
            if d.get("class_name") == GAME_ID_CLASS_NAME:
                # quick OCR
                val = None
                try:
                    val = self.ocr_crop(frame, {"x1":d["x1"],"y1":d["y1"],"x2":d["x2"],"y2":d["y2"]})
                except Exception:
                    val = None
                if val:
                    game_ids.append(val)
        if game_ids:
            gid = game_ids[-1]
            if self.last_game_id is None:
                self.last_game_id = gid
            elif gid != self.last_game_id:
                # new game started
                print("[GAMEID CHANGE] -> finalizing current hand")
                self.finalize_hand(buttons, input_field)
                self.last_game_id = gid
        # finalize by timeout
        if self.hand_start_ts and self.last_button_seen_ts:
            if time.time() - self.last_button_seen_ts > HAND_TIMEOUT:
                print("[HAND TIMEOUT] finalizing due to inactivity")
                self.finalize_hand(buttons, input_field)

        # 5) debug drawing
        debug = frame.copy()
        for t in active:
            tid = t["track_id"]
            votes = list(self.track_history[tid]["class_votes"])
            cls = Counter(votes).most_common(1)[0][0] if votes else t["class"]
            avg = avg_bbox(self.track_history[tid]["bboxes"])
            x1,y1,x2,y2 = avg
            cv2.rectangle(debug, (x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(debug, f"ID{tid}:{cls}", (x1, max(0,y1-6)), FONT, 0.45, (0,255,0),1,cv2.LINE_AA)
        if SHOW_DEBUG_WINDOW:
            status = f"Hand#{self.hand_index} active:{len(buttons)} buttons"
            if self.hand_start_ts:
                status += " | IN-HAND"
            cv2.putText(debug, status, (10,20), FONT, 0.6, (0,255,255),2,cv2.LINE_AA)
            cv2.imshow("Poker Debug", cv2.resize(debug, (min(debug.shape[1], 1200), int(debug.shape[0]*min(1,1200/debug.shape[1])))))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False, {"active": active, "buttons": buttons, "input": input_field}
        return True, {"active": active, "buttons": buttons, "input": input_field}

# ----------------- Main loop -----------------

def main_loop():
    pipeline = LivePokerPipeline(MODEL_PATH)
    print("Pipeline init done. SAHI available:", SAHI_AVAILABLE and pipeline.sahi_model is not None, "DeepSORT available:", DEEPSORT_AVAILABLE)
    sct = mss.mss()
    region = SCREEN_REGION
    frame_interval = 1.0 / max(1, INPUT_FPS)
    last = 0
    try:
        while True:
            t0 = time.time()
            if t0 - last < frame_interval:
                time.sleep(max(0, frame_interval - (t0 - last)))
            last = time.time()
            img = np.array(sct.grab(region))
            # mss returns BGRA
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cont, info = pipeline.step(frame)
            # allow exit if cont False or window closed
            if not cont:
                break
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if SHOW_DEBUG_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
