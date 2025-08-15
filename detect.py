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
RANK_MODEL_PATH = "rank_classifier.pt"  # Added rank classifier model path
OUTPUT_FOLDER = "live_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SCREEN_REGION = {"top": 48, "left": 970, "width": 930, "height": 1130}

model = YOLO(MODEL_PATH)  # Main poker detection model
rank_model = YOLO(RANK_MODEL_PATH)  # Rank classification model

tracker = sv.ByteTrack()
use_gpu_for_ocr = torch.cuda.is_available()
ocr_reader = easyocr.Reader(['en'], gpu=use_gpu_for_ocr)

JSON_OUTPUT = os.path.join(OUTPUT_FOLDER, "detections.json")
all_detections = []

bbox_annotator = sv.BoxAnnotator()

VALID_CARD_RANKS = {'A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'}
MAPPING_CORRECTION = {'O': '0', 'I': '1', 'S': '5', 'Z': '2', 'B': '8'}

RANK_CLASSES = {'card1_rank', 'card2_rank', 'flop1_rank', 'flop2_rank', 'flop3_rank', 'turn_rank', 'river_rank'}

def safe_crop(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def classify_card_rank(crop):
    """Use YOLO rank classifier to detect card rank"""
    if crop is None or crop.size == 0:
        return ""
    
    try:
        # Run rank classification on the cropped region
        rank_results = rank_model(crop)[0]
        
        # Classification models return .probs, not detections with bounding boxes
        if hasattr(rank_results, 'probs') and rank_results.probs is not None:
            # Get the top prediction
            top_class_id = rank_results.probs.top1  # Index of highest confidence class
            top_confidence = rank_results.probs.top1conf.item()  # Confidence score
            
            # Get rank name from model
            rank_name = rank_model.names.get(top_class_id, "")
            
            # Only return if confidence is high enough
            if top_confidence > 0.5:  # Adjust threshold as needed
                print(f"[v0] Rank classified: {rank_name} (confidence: {top_confidence:.3f})")
                return rank_name
            else:
                print(f"[v0] Low confidence rank: {rank_name} (confidence: {top_confidence:.3f})")
        
        return ""
    except Exception as e:
        print(f"Rank classification error: {e}")
        return ""

def enhance_for_ocr(image, enhancement_type="standard"):
    """Enhanced preprocessing for better OCR accuracy"""
    if image is None or image.size == 0:
        return image
    
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if enhancement_type == "card_rank":
            # Specialized processing for card ranks (small text)
            # 1. Upscale 3x for better OCR
            height, width = gray.shape
            gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            
            # 2. CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 3. Denoising
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            
            # 4. Sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
            
            # 5. Adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # 6. Morphological operations to clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return binary
            
        elif enhancement_type == "game_id":
            # Processing for game IDs (usually larger but may be low contrast)
            # 1. Upscale 2x
            height, width = gray.shape
            gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            
            # 2. Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 3. Edge enhancement
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            gray = cv2.filter2D(gray, -1, kernel)
            
            # 4. Binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
            
        else:  # standard enhancement
            # Basic enhancement for other elements
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return enhanced
            
    except Exception as e:
        print(f"Enhancement error: {e}")
        return image

def extract_text_with_multiple_methods(crop, class_name):
    """Try multiple OCR methods for better accuracy with focus on card ranks"""
    if crop is None:
        return ""
    
    if class_name.lower() in RANK_CLASSES:
        return classify_card_rank(crop)
    
    best_text = ""
    best_confidence = 0
    
    # Determine enhancement type based on class name
    enhancement_type = "game_id" if "game_id" in class_name.lower() else "standard"

    try:
        enhanced_crop = enhance_for_ocr(crop, enhancement_type)
        if enhanced_crop is not None:
            ocr_results = ocr_reader.readtext(enhanced_crop, detail=1, paragraph=False)
            for bbox, text, conf in ocr_results:
                if conf > best_confidence:
                    best_text = text.strip()
                    best_confidence = conf
        
        if best_confidence < 0.7:
            # Method 2: Different thresholding
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_results = ocr_reader.readtext(thresh, detail=1, paragraph=False)
            for bbox, text, conf in ocr_results:
                if conf > best_confidence:
                    best_text = text.strip()
                    best_confidence = conf
        
        if best_confidence < 0.6:
            # Method 3: Scaling
            resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ocr_results = ocr_reader.readtext(resized, detail=1, paragraph=False)
            for bbox, text, conf in ocr_results:
                if conf > best_confidence:
                    best_text = text.strip()
                    best_confidence = conf
        
        return best_text if best_confidence > 0.4 else ""
        
    except Exception as e:
        print(f"OCR error for {class_name}: {e}")
        return ""

def create_clean_detections(xyxy, class_id=None, confidence=None, tracker_id=None):
    """Create a completely clean Detections object with guaranteed valid data types"""
    if len(xyxy) == 0:
        return sv.Detections.empty()
    
    # Ensure xyxy is proper format
    xyxy = np.array(xyxy, dtype=np.float32)
    
    # Handle class_id - ensure all are valid integers
    if class_id is None:
        class_id = np.zeros(len(xyxy), dtype=np.int32)
    else:
        clean_class_ids = []
        for cid in class_id:
            if cid is None or np.isnan(float(cid)) if isinstance(cid, (int, float)) else False:
                clean_class_ids.append(0)  # Use 0 instead of -1 to avoid negative index issues
            else:
                clean_class_ids.append(int(cid))
        class_id = np.array(clean_class_ids, dtype=np.int32)
    
    # Handle confidence
    if confidence is None:
        confidence = np.ones(len(xyxy), dtype=np.float32)
    else:
        clean_confidence = []
        for conf in confidence:
            if conf is None or np.isnan(float(conf)) if isinstance(conf, (int, float)) else False:
                clean_confidence.append(0.0)
            else:
                clean_confidence.append(float(conf))
        confidence = np.array(clean_confidence, dtype=np.float32)
    
    # Handle tracker_id
    if tracker_id is not None:
        clean_tracker_ids = []
        for tid in tracker_id:
            if tid is None:
                clean_tracker_ids.append(-1)
            else:
                clean_tracker_ids.append(int(tid))
        tracker_id = np.array(clean_tracker_ids, dtype=np.int32)
    
    # Create new detections object
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        tracker_id=tracker_id
    )
    
    return detections

with mss.mss() as sct:
    frame_count = 0
    print("Starting live detection. Press 'q' in the window to quit.")
    while True:
        start_time = time.time()

        # Capture screen
        screenshot = np.array(sct.grab(SCREEN_REGION))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # YOLO detection
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Skip if no detections
        if len(detections.xyxy) == 0:
            cv2.imshow("Live Detection + OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Create clean detections object
        detections = create_clean_detections(
            detections.xyxy,
            detections.class_id,
            detections.confidence
        )

        # Track objects
        tracked = tracker.update_with_detections(detections)

        dets_for_annotate = None
        labels = []
        frame_data = []

        # If tracker returned detections
        if isinstance(tracked, sv.Detections) and len(tracked.xyxy) > 0:
            # Create clean tracked detections
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

                crop = safe_crop(frame, x1, y1, x2, y2)
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

                crop = safe_crop(frame, x1, y1, x2, y2)
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

        # Final safety check - recreate detections one more time to be absolutely sure
        if len(dets_for_annotate.xyxy) > 0:
            dets_for_annotate = create_clean_detections(
                dets_for_annotate.xyxy,
                dets_for_annotate.class_id,
                dets_for_annotate.confidence,
                dets_for_annotate.tracker_id
            )

            # Match labels length to number of boxes
            num_boxes = len(dets_for_annotate.xyxy)
            if len(labels) < num_boxes:
                labels += [""] * (num_boxes - len(labels))
            elif len(labels) > num_boxes:
                labels = labels[:num_boxes]

            # Annotate safely
            try:
                annotated_frame = bbox_annotator.annotate(frame.copy(), dets_for_annotate, labels)
            except Exception as e:
                print(f"Annotation error: {e}")
                print(f"class_id values: {dets_for_annotate.class_id}")
                annotated_frame = frame.copy()
        else:
            annotated_frame = frame.copy()

        # Save frame
        frame_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, annotated_frame)

        # Save JSON
        all_detections.append({"frame": frame_count, "timestamp": time.time(), "detections": frame_data})
        with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(all_detections, f, indent=2)

        cv2.imshow("Live Detection + OCR", annotated_frame)

        frame_count += 1
        fps = 1.0 / (time.time() - start_time + 1e-6)
        print(f"Frame {frame_count} | FPS: {fps:.2f} | Detections: {len(frame_data)}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
