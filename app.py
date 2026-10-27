"""
Automatic Number Plate Recognition (ANPR)
===========================================
A clean, modern Computer Vision web application built with Streamlit, OpenCV, and EasyOCR.
Detects and extracts vehicle registration numbers, with dedicated support for:
- Standard Indian Registration Plates (e.g., MH 12 AB 1234, DL 3C AB 1234)
- Bharat Series (BH) Plates (e.g., 22 BH 1234 AA)
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image
import streamlit as st

# ---------------------------------------------------------
# Indian State & Union Territory Codes Mapping
# ---------------------------------------------------------
INDIAN_STATES: Dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman & Diu",
    "DL": "Delhi",
    "DN": "Dadra & Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu & Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# ---------------------------------------------------------
# Cache EasyOCR Reader to avoid reloading weights each run
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ocr_reader() -> easyocr.Reader:
    """Loads and caches EasyOCR reader instance for English characters."""
    return easyocr.Reader(["en"], gpu=False)


# ---------------------------------------------------------
# Text Cleaning & Indian Number Plate Regex Matching
# ---------------------------------------------------------
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}

DIGIT_TO_LETTER = {
    "0": "D",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
}

LETTER_TO_DIGIT = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "l": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "q": "9",
}


def clean_raw_text(raw_text: str) -> str:
    """Cleans raw OCR output by removing non-alphanumeric characters."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
    # Strip HSRP 'IND' prefix if present before registration string
    if cleaned.startswith("IND") and len(cleaned) >= 8:
        cleaned = cleaned[3:]
    return cleaned


def parse_and_validate_plate(raw_text: str) -> Optional[Dict[str, str]]:
    """
    Validates and formats text against Indian vehicle registration patterns:
    1. Standard Indian Format: [State 2L][District 1-2D][Series 0-3L][Number 4D]
       e.g., MH12AB1234, DL3CAB1234, GJ08DG9292, KA05M9876
    2. Bharat Series (BH): [Year 2D][BH][Number 4D][Series 1-2L]
       e.g., 22BH1234AA
    """
    cleaned = clean_raw_text(raw_text)
    if len(cleaned) < 6:
        return None

    # Helper function to attempt standard format validation
    def check_standard(candidate: str, raw_input: str = "") -> Optional[Dict[str, str]]:
        if raw_input:
            spaced_clean = re.sub(r"[^A-Za-z0-9\s]", "", raw_input).upper().strip()
            if spaced_clean.startswith("IND"):
                spaced_clean = spaced_clean[3:].strip()
            m_spaced = re.match(r"^([A-Z]{2})\s+([0-9]{1,2}[A-Z]?)\s+([A-Z]{1,3})\s+([0-9]{4})$", spaced_clean)
            if m_spaced:
                state, district, series, number = m_spaced.groups()
                if state not in INDIAN_STATES:
                    return None
                state_name = INDIAN_STATES[state]
                return {
                    "plate_number": f"{state} {district} {series} {number}",
                    "clean_text": candidate,
                    "type": "Standard Indian Plate",
                    "state_code": state,
                    "state_name": state_name,
                    "district_code": district,
                    "series": series,
                    "unique_number": number,
                }

        # Delhi-specific format in unspaced string: DL + 1-2 digits + 1 vehicle category letter + series + 4 digits
        m_delhi = re.match(r"^(DL)([0-9]{1,2})([A-Z])([A-Z]{1,2})([0-9]{4})$", candidate)
        if m_delhi:
            state, dist_num, cat_letter, series, number = m_delhi.groups()
            district = f"{dist_num}{cat_letter}"
            return {
                "plate_number": f"{state} {district} {series} {number}",
                "clean_text": candidate,
                "type": "Standard Indian Plate",
                "state_code": state,
                "state_name": "Delhi",
                "district_code": district,
                "series": series,
                "unique_number": number,
            }

        # General standard Indian plates:
        # State: 2 letters, District: 1 or 2 digits, Series: 0 to 3 letters, Number: 4 digits
        pattern = r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$"
        m = re.match(pattern, candidate)
        if m:
            state, district, series, number = m.groups()
            if state not in INDIAN_STATES:
                return None
            state_name = INDIAN_STATES[state]
            formatted = f"{state} {district} {series} {number}".replace("  ", " ").strip()
            return {
                "plate_number": formatted,
                "clean_text": candidate,
                "type": "Standard Indian Plate",
                "state_code": state,
                "state_name": state_name,
                "district_code": district,
                "series": series if series else "None",
                "unique_number": number,
            }
        return None

    # Helper function to attempt Indian Military / Armed Forces plate validation
    def check_military(candidate: str) -> Optional[Dict[str, str]]:
        # Pattern for Indian Armed Forces / Military plate:
        # e.g., ↑ 20B 132976 N (or OCR-ed as T208132976N, ^20B132976N, 120B132976N, 20B132976N)
        m = re.match(r"^([T\^1I↑]?)([0-9]{2})([A-Z0-9])([0-9]{6})([A-Z])$", candidate)
        if m:
            arrow, year, cls_c, serial, check = m.groups()
            if cls_c in ["8", "B"]:
                cls_c = "B"
            elif cls_c in ["0", "D"]:
                cls_c = "D"
            elif cls_c in ["1", "I"]:
                cls_c = "A"

            class_desc = {
                "A": "Two-Wheeler",
                "B": "Car / Light Motor Vehicle (Jeep/Sedan)",
                "C": "Light Truck / 1-Ton",
                "D": "Medium Truck / Heavy Transport",
                "E": "Specialized Combat Vehicle",
            }.get(cls_c, f"Class {cls_c}")

            formatted = f"↑ {year}{cls_c} {serial} {check}"
            return {
                "plate_number": formatted,
                "clean_text": candidate,
                "type": "Indian Military / Defence Plate (Armed Forces)",
                "procurement_year": f"20{year}",
                "vehicle_class": f"Class {cls_c} ({class_desc})",
                "serial_number": serial,
                "check_code": check,
                "state_name": "Ministry of Defence, Govt of India (Indian Army / Armed Forces)",
            }
        return None

    # Helper function to attempt BH-series validation
    def check_bh(candidate: str) -> Optional[Dict[str, str]]:
        pattern = r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$"
        m = re.match(pattern, candidate)
        if m:
            year, code, number, series = m.groups()
            formatted = f"{year} {code} {number} {series}"
            return {
                "plate_number": formatted,
                "clean_text": candidate,
                "type": "Bharat Series (BH)",
                "year": f"20{year}",
                "series": series,
                "unique_number": number,
                "state_name": "Pan-India Central Registration (BH Series)",
            }
        return None

    # Sliding window candidates to handle text with surrounding noise/watermarks
    candidates_to_test = []
    if 8 <= len(cleaned) <= 12:
        candidates_to_test.append(cleaned)
    elif len(cleaned) > 12:
        candidates_to_test.append(cleaned)
        for w in [11, 10, 9, 8]:
            for i in range(len(cleaned) - w + 1):
                candidates_to_test.append(cleaned[i : i + w])

    for cand in candidates_to_test:
        # A valid Indian plate candidate must contain at least 3 digits (district + vehicle number)
        if sum(c.isdigit() for c in cand) < 3:
            continue

        # 1. Direct regex check (Military, BH Series, or Standard State plate)
        res = check_military(cand) or check_bh(cand) or check_standard(cand, raw_text)
        if res:
            return res

        # 2. Heuristic correction for 10-character Standard Plate:
        # Position 0-1: State (Letters)
        # Position 2-3: District (Digits)
        # Position 4-5: Series (Letters - e.g. OCR turns D->0, G->6, B->8)
        # Position 6-9: Number (Digits)
        if len(cand) == 10:
            ch = list(cand)
            ch[0] = DIGIT_TO_LETTER.get(ch[0], ch[0])
            ch[1] = DIGIT_TO_LETTER.get(ch[1], ch[1])
            ch[2] = LETTER_TO_DIGIT.get(ch[2], ch[2])
            ch[3] = LETTER_TO_DIGIT.get(ch[3], ch[3])
            ch[4] = DIGIT_TO_LETTER.get(ch[4], ch[4])
            ch[5] = DIGIT_TO_LETTER.get(ch[5], ch[5])
            for i in range(6, 10):
                ch[i] = LETTER_TO_DIGIT.get(ch[i], ch[i])
            res = check_standard("".join(ch), raw_text)
            if res:
                return res

        # 3. Heuristic correction for 9-character Standard Plate:
        if len(cand) == 9:
            # Variant A: 1-digit district, 2-letter series (e.g. MH 1 AA 2345)
            ch1 = list(cand)
            ch1[0] = DIGIT_TO_LETTER.get(ch1[0], ch1[0])
            ch1[1] = DIGIT_TO_LETTER.get(ch1[1], ch1[1])
            ch1[2] = LETTER_TO_DIGIT.get(ch1[2], ch1[2])
            ch1[3] = DIGIT_TO_LETTER.get(ch1[3], ch1[3])
            ch1[4] = DIGIT_TO_LETTER.get(ch1[4], ch1[4])
            for i in range(5, 9):
                ch1[i] = LETTER_TO_DIGIT.get(ch1[i], ch1[i])
            res1 = check_standard("".join(ch1), raw_text)
            if res1:
                return res1

            # Variant B: 2-digit district, 1-letter series (e.g. MH 12 A 1234)
            ch2 = list(cand)
            ch2[0] = DIGIT_TO_LETTER.get(ch2[0], ch2[0])
            ch2[1] = DIGIT_TO_LETTER.get(ch2[1], ch2[1])
            ch2[2] = LETTER_TO_DIGIT.get(ch2[2], ch2[2])
            ch2[3] = LETTER_TO_DIGIT.get(ch2[3], ch2[3])
            ch2[4] = DIGIT_TO_LETTER.get(ch2[4], ch2[4])
            for i in range(5, 9):
                ch2[i] = LETTER_TO_DIGIT.get(ch2[i], ch2[i])
            res2 = check_standard("".join(ch2), raw_text)
            if res2:
                return res2

        # 4. Heuristic correction for BH-series plates (9 or 10 characters):
        if len(cand) in [9, 10]:
            ch_bh = list(cand)
            ch_bh[0] = LETTER_TO_DIGIT.get(ch_bh[0], ch_bh[0])
            ch_bh[1] = LETTER_TO_DIGIT.get(ch_bh[1], ch_bh[1])
            ch_bh[2] = DIGIT_TO_LETTER.get(ch_bh[2], ch_bh[2])
            ch_bh[3] = DIGIT_TO_LETTER.get(ch_bh[3], ch_bh[3])
            for i in range(4, 8):
                ch_bh[i] = LETTER_TO_DIGIT.get(ch_bh[i], ch_bh[i])
            for i in range(8, len(ch_bh)):
                ch_bh[i] = DIGIT_TO_LETTER.get(ch_bh[i], ch_bh[i])
            res_bh = check_bh("".join(ch_bh))
            if res_bh:
                return res_bh

    return None


# ---------------------------------------------------------
# OpenCV Image Preprocessing & Plate Localization
# ---------------------------------------------------------
def preprocess_image(image_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Performs standard OpenCV preprocessing steps:
    1. Grayscale conversion
    2. Bilateral filtering for edge-preserving noise reduction
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Canny Edge Detection
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    smoothed = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(smoothed)
    edges = cv2.Canny(contrast_enhanced, 30, 200)

    return {
        "gray": gray,
        "smoothed": smoothed,
        "enhanced": contrast_enhanced,
        "edges": edges,
    }


def find_plate_candidates(
    image_rgb: np.ndarray, edges: np.ndarray
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Finds candidate license plate rectangular contours based on aspect ratio and area.
    Returns a list of tuples: (cropped_rgb_image, (x, y, w, h))
    """
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    candidates = []
    img_h, img_w = image_rgb.shape[:2]
    min_area = (img_h * img_w) * 0.001
    max_area = (img_h * img_w) * 0.50

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h)

        if 1.5 <= aspect_ratio <= 6.5:
            pad_x = int(w * 0.08)
            pad_y = int(h * 0.08)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)

            crop = image_rgb[y1:y2, x1:x2]
            candidates.append((crop, (x1, y1, x2 - x1, y2 - y1)))

    return candidates


# ---------------------------------------------------------
# OCR Extraction Pipeline
# ---------------------------------------------------------
def detect_number_plate(
    image_rgb: np.ndarray,
    reader: easyocr.Reader,
    min_confidence: float = 0.20,
) -> Tuple[Optional[Dict[str, str]], np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Runs the full ANPR pipeline:
    1. Preprocesses image with OpenCV.
    2. Identifies rectangular license plate candidates and analyzes full image.
    3. Uses EasyOCR with regex validation to prioritize valid Indian vehicle numbers.
    4. Draws bounding box and returns annotated image and metadata.
    """
    preprocessed = preprocess_image(image_rgb)
    annotated_image = image_rgb.copy()
    plate_crop = None
    best_match: Optional[Dict[str, str]] = None
    best_bbox = None
    best_confidence = 0.0
    img_h, img_w = image_rgb.shape[:2]

    # Strategy 1: Check EasyOCR detections on full image
    full_ocr = reader.readtext(image_rgb)
    for bbox, text, conf in full_ocr:
        if conf < min_confidence:
            continue

        match = parse_and_validate_plate(text)
        if match and conf > best_confidence:
            best_match = match
            best_confidence = float(conf)
            pts = np.array(bbox, dtype=np.int32)
            x = int(np.min(pts[:, 0]))
            y = int(np.min(pts[:, 1]))
            w = int(np.max(pts[:, 0]) - x)
            h = int(np.max(pts[:, 1]) - y)
            best_bbox = (x, y, w, h)
            x1, y1 = max(0, x - 5), max(0, y - 5)
            x2, y2 = min(img_w, x + w + 5), min(img_h, y + h + 5)
            plate_crop = image_rgb[y1:y2, x1:x2]

    # Strategy 2: Check pairs of OCR detections (for two-line or split plates)
    if not best_match and len(full_ocr) >= 2:
        for i in range(len(full_ocr)):
            for j in range(len(full_ocr)):
                if i == j:
                    continue
                combined_pair = full_ocr[i][1] + " " + full_ocr[j][1]
                pair_match = parse_and_validate_plate(combined_pair)
                if pair_match:
                    pair_conf = float((full_ocr[i][2] + full_ocr[j][2]) / 2.0)
                    if pair_conf > best_confidence:
                        best_match = pair_match
                        best_confidence = pair_conf
                        all_pts = np.concatenate([np.array(full_ocr[i][0], dtype=np.int32), np.array(full_ocr[j][0], dtype=np.int32)])
                        x = int(np.min(all_pts[:, 0]))
                        y = int(np.min(all_pts[:, 1]))
                        w = int(np.max(all_pts[:, 0]) - x)
                        h = int(np.max(all_pts[:, 1]) - y)
                        best_bbox = (x, y, w, h)
                        plate_crop = image_rgb[max(0, y - 5):min(img_h, y + h + 5), max(0, x - 5):min(img_w, x + w + 5)]
                        break
            if best_match:
                break

    # Strategy 3: OCR on localized candidate contour crops with upscaling
    if not best_match:
        candidates = find_plate_candidates(image_rgb, preprocessed["edges"])
        for crop, (cx, cy, cw, ch) in candidates:
            # Upscale crop if small to improve EasyOCR accuracy
            crop_to_ocr = crop
            if cw < 200 or ch < 60:
                scale = max(200 / max(cw, 1), 60 / max(ch, 1))
                crop_to_ocr = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            crop_ocr = reader.readtext(crop_to_ocr)
            for _, c_text, c_conf in crop_ocr:
                c_match = parse_and_validate_plate(c_text)
                if c_match and c_conf > best_confidence:
                    best_match = c_match
                    best_confidence = float(c_conf)
                    best_bbox = (cx, cy, cw, ch)
                    plate_crop = crop
                    break

            if not best_match and crop_ocr:
                combined_crop_text = " ".join([entry[1] for entry in crop_ocr])
                c_match = parse_and_validate_plate(combined_crop_text)
                if c_match:
                    avg_c_conf = float(np.mean([entry[2] for entry in crop_ocr]))
                    if avg_c_conf >= min_confidence:
                        best_match = c_match
                        best_confidence = avg_c_conf
                        best_bbox = (cx, cy, cw, ch)
                        plate_crop = crop
                        break

    # Strategy 4: If still not found, run on CLAHE contrast enhanced image
    if not best_match:
        enhanced_rgb = cv2.cvtColor(preprocessed["enhanced"], cv2.COLOR_GRAY2RGB)
        enh_ocr = reader.readtext(enhanced_rgb)
        for bbox, text, conf in enh_ocr:
            match = parse_and_validate_plate(text)
            if match and conf > best_confidence:
                best_match = match
                best_confidence = float(conf)
                pts = np.array(bbox, dtype=np.int32)
                x = int(np.min(pts[:, 0]))
                y = int(np.min(pts[:, 1]))
                w = int(np.max(pts[:, 0]) - x)
                h = int(np.max(pts[:, 1]) - y)
                best_bbox = (x, y, w, h)
                x1, y1 = max(0, x - 5), max(0, y - 5)
                x2, y2 = min(img_w, x + w + 5), min(img_h, y + h + 5)
                plate_crop = image_rgb[y1:y2, x1:x2]
                break

    # Draw bounding box and label if plate is detected
    if best_match and best_bbox is not None:
        x, y, w, h = best_bbox
        best_match["confidence"] = f"{best_confidence * 100:.1f}%"

        # Draw aesthetic high-visibility bounding box
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        label = f"{best_match['plate_number']} ({best_match['confidence']})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y - 10, text_h + 10)
        # Background tag
        cv2.rectangle(
            annotated_image,
            (x, label_y - text_h - 6),
            (x + text_w + 10, label_y + baseline - 2),
            (0, 200, 0),
            -1,
        )
        # Label text
        cv2.putText(
            annotated_image,
            label,
            (x + 5, label_y - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return best_match, annotated_image, plate_crop, preprocessed


# ---------------------------------------------------------
# Streamlit Modern UI
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Automatic Number Plate Recognition (ANPR)",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Modern Custom CSS styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .plate-container {
            background: linear-gradient(135deg, #0F172A, #1E293B);
            border-radius: 12px;
            padding: 22px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 15px 0;
            border-left: 6px solid #10B981;
        }
        .plate-box {
            display: inline-flex;
            align-items: center;
            background: #FFFFFF;
            color: #000000;
            font-family: 'Courier New', Courier, monospace;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 3px;
            padding: 8px 24px;
            border-radius: 8px;
            border: 3px solid #000000;
            box-shadow: 0 3px 6px rgba(0,0,0,0.2);
            margin: 10px 0;
        }
        .ind-badge {
            background: #003399;
            color: white;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 4px;
            margin-right: 12px;
            display: inline-block;
        }
        .metric-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<div class="main-header">🚗 Automatic Number Plate Recognition (ANPR)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Computer Vision application powered by OpenCV & EasyOCR. Optimized for Indian Standard & BH-Series license plates.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        min_conf = st.slider("OCR Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
        show_steps = st.checkbox("Show OpenCV Preprocessing Steps", value=True)

        st.markdown("---")
        st.subheader("💡 Supported Formats")
        st.markdown(
            """
            - **Standard Indian**: `MH 12 AB 1234`
              *(State Code + District Code + Series + 4 Digits)*
            - **Bharat (BH) Series**: `22 BH 1234 AA`
              *(Year + BH + 4 Digits + Series)*
            - **Auto-corrections**:
              Cleans HSRP 'IND' stamps and corrects common digit/letter confusions.
            """
        )

        st.markdown("---")
        st.subheader("📁 Test with Sample Images")
        sample_folder = "sample_images"
        sample_options = ["None (Upload Own Image)"]
        if os.path.exists(sample_folder):
            samples = [f for f in os.listdir(sample_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            sample_options.extend(sorted(samples))

        selected_sample = st.selectbox("Select a Sample Vehicle Image", sample_options)

    # Main Area: Upload or Sample Selection
    image_input = None
    uploaded_file = st.file_uploader(
        "Upload a vehicle image (front or rear view with license plate)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a car or vehicle",
    )

    if uploaded_file is not None:
        image_input = Image.open(uploaded_file).convert("RGB")
    elif selected_sample != "None (Upload Own Image)":
        sample_path = os.path.join(sample_folder, selected_sample)
        if os.path.exists(sample_path):
            image_input = Image.open(sample_path).convert("RGB")
            st.info(f"Loaded sample image: **{selected_sample}**")

    if image_input is None:
        st.info("👆 Please upload a vehicle image or select a sample image from the sidebar to begin recognition.")
        return

    # Convert PIL Image to RGB NumPy array
    image_np = np.array(image_input)

    # Initialize OCR model
    with st.spinner("Initializing EasyOCR Engine..."):
        reader = load_ocr_reader()

    # Process image
    with st.spinner("Processing image and recognizing number plate..."):
        result, annotated_img, plate_crop, preproc = detect_number_plate(
            image_np, reader, min_confidence=min_conf
        )

    # Results Display Layout
    st.markdown("---")
    col1, col2 = st.columns([1.1, 1.0], gap="large")

    with col1:
        st.subheader("📷 Image Preview")
        st.image(
            annotated_img if result else image_np,
            caption="Processed Image with Detection Box" if result else "Original Uploaded Image",
            width="stretch",
        )

    with col2:
        st.subheader("🔍 Extraction Results")
        if result:
            badge_label = "ARMY" if "Military" in result.get("type", "") else "IND"
            badge_color = "#2D5A27" if "Military" in result.get("type", "") else "#003399"
            st.markdown(
                f"""
                <div class="plate-container">
                    <div style="font-size: 0.9rem; text-transform: uppercase; color: #94A3B8; margin-bottom: 5px;">
                        Detected Number Plate
                    </div>
                    <div class="plate-box">
                        <span class="ind-badge" style="background: {badge_color};">{badge_label}</span>
                        {result['plate_number']}
                    </div>
                    <div style="margin-top: 12px; font-size: 0.95rem; color: #E2E8F0;">
                        <b>Format Type:</b> {result['type']} &nbsp;|&nbsp; <b>Confidence:</b> {result['confidence']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Details Breakdown
            st.markdown("#### 📋 Registration Details")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Authority / State:** {result.get('state_name', 'N/A')}")
                if "procurement_year" in result:
                    st.markdown(f"**Procurement Year:** `{result['procurement_year']}`")
                elif "district_code" in result:
                    st.markdown(f"**RTO / District:** `{result['district_code']}`")
                elif "year" in result:
                    st.markdown(f"**Registration Year:** `{result['year']}`")
            with c2:
                if "vehicle_class" in result:
                    st.markdown(f"**Vehicle Class:** `{result['vehicle_class']}`")
                    st.markdown(f"**Serial / Check:** `{result.get('serial_number', 'N/A')} ({result.get('check_code', 'N/A')})`")
                else:
                    st.markdown(f"**Series:** `{result.get('series', 'N/A')}`")
                    st.markdown(f"**Unique Number:** `{result.get('unique_number', 'N/A')}`")

            if plate_crop is not None and plate_crop.size > 0:
                st.markdown("#### 🔎 Cropped Plate Region")
                st.image(plate_crop, caption="Cropped License Plate", width=320)

        else:
            st.warning(
                "⚠️ **No valid Indian vehicle number plate was detected.**\n\n"
                "**Troubleshooting Tips:**\n"
                "- Ensure the license plate is clearly visible and reasonably lit.\n"
                "- Avoid extreme angles or excessive glare on the plate.\n"
                "- Try lowering the **OCR Confidence Threshold** slider in the sidebar.\n"
                "- Supported formats: Standard State Plates (e.g. MH 12 AB 1234), BH Series (22 BH 1234 AA), and Indian Military / Defence plates (↑ 20B 132976 N)."
            )

    # Preprocessing Pipeline Visualization
    if show_steps and preproc:
        st.markdown("---")
        st.subheader("🛠️ OpenCV Image Preprocessing Pipeline")
        st.caption("Step-by-step transformation applied to prepare the image for OCR:")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.image(preproc["gray"], caption="1. Grayscale Conversion", width="stretch")
        with p_col2:
            st.image(preproc["smoothed"], caption="2. Bilateral Filter (Noise Reduction)", width="stretch")
        with p_col3:
            st.image(preproc["enhanced"], caption="3. CLAHE (Contrast Enhancement)", width="stretch")
        with p_col4:
            st.image(preproc["edges"], caption="4. Canny Edge Detection", width="stretch")


if __name__ == "__main__":
    main()
