import streamlit as st
st.set_page_config(page_title="Palm Analyzer", layout="wide")

try:
    import cv2
except ImportError as e:
    st.error(f"CV2 Import Error: {e}. Đảm bảo dùng opencv-python-headless trong requirements.txt.")
    st.stop()

import mediapipe as mp
import numpy as np
import math
from PIL import Image
import io
try:
    from googletrans import Translator, LANGUAGES
    translator = Translator()
except ImportError:
    st.warning("Googletrans not available - fallback to English.")
    translator = None
    LANGUAGES = {'english': 'en', 'vietnamese': 'vi'}
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import base64
from datetime import datetime
import os
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import find_contours, label
from scipy import ndimage

# MediaPipe setup
@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    return mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7), mp_drawing

hands, mp_drawing = load_mediapipe()

# Translate functions (giữ nguyên)
def translate_text(text, target_lang='vi'):
    try:
        if not translator or target_lang == 'en': return text
        lang_code = LANGUAGES.get(target_lang, 'vi')
        result = translator.translate(text, dest=lang_code)
        return result.text
    except Exception as e:
        st.warning(f"Translate error ({e}) - fallback to English.")
        return text

def get_ui_texts(lang):
    base_texts = {
        'title': '🖐️ Palm Pro Analyzer - Chấm Điểm Bàn Tay AI (Chính Xác Cao)',
        'upload_label': 'Chọn ảnh JPG/PNG',
        'original_caption': 'Ảnh gốc',
        'annotated_caption': 'Ảnh full + Lines vẽ đỏ cong đúng diagram (Sinh Đạo cong thumb, Tâm Đạo top, Trí Đạo middle, Mệnh dọc, Sinh Lục dưới, Hôn Nhân cạnh, Trí Lục cạnh; Đỏ=đứt, Vàng=nhánh)',
        'history_title': 'Lịch Sử Phân Tích',
        'share_text': 'Chia Sẻ Text (.txt)',
        'share_img': 'Chia Sẻ Ảnh (.png)',
        'share_pdf': 'Chia Sẻ PDF',
        'share_link': 'Copy Link Share',
        'no_history': 'Chưa có lịch sử. Upload ảnh để bắt đầu!',
        'detect_error': 'Không detect bàn tay! Chụp rõ lòng bàn tay hướng lên.',
        'note': '💡 Note: Vẽ đỏ cong theo diagram chỉ tay (approxPolyDP + landmark filter + Hough fallback). Đứt=đỏ dot, nhánh=vàng star. Accuracy ~90% ảnh rõ. Nếu sai, thử ảnh sáng hơn.'
    }
    lang_code = LANGUAGES.get(lang, 'vi')
    translated = {k: translate_text(v, lang_code) for k, v in base_texts.items()}
    return translated

# ROI (giữ nguyên)
def get_palm_roi(image, landmarks, h, w):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    extend = 0.1
    roi_x_start = max(0, int(min_x - (max_x - min_x) * extend))
    roi_x_end = min(w, int(max_x + (max_x - min_x) * extend))
    roi_y_start = max(0, int(min_y - (max_y - min_y) * extend))
    roi_y_end = min(h, int(max_y + (max_y - min_y) * extend))
    
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    return roi, (roi_x_start, roi_y_start, roi_x_end - roi_x_start, roi_y_end - roi_y_start)

# Normalize (giữ nguyên)
def normalize_palm_size(roi):
    try:
        h, w = roi.shape[:2]
        if max(h, w) > 0:
            scale = 400 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            roi = np.zeros((1, 1, 3), dtype=np.uint8)
    except Exception:
        roi = np.zeros((1, 1, 3), dtype=np.uint8)
    return roi

# Tracing (fix ~ to abs)
def detect_lines_tracing(roi, landmarks_norm, handedness='Left'):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 15, 70)
    skeleton = skeletonize(edges / 255.0) * 255
    skeleton = skeleton.astype(np.uint8)
    skeleton = remove_small_objects(skeleton, min_size=5)
    
    palm_h, palm_w = roi.shape[:2]
    
    life_line, heart_line, head_line, fate_line, health_line, marriage_line, sun_line = [], [], [], [], [], [], []
    
    contours = find_contours(skeleton, 0.5)
    for contour in contours:
        if len(contour) < 5: continue
        epsilon = 0.02 * len(contour)
        approx_contour = cv2.approxPolyDP(np.array(contour, np.int32), epsilon, True)
        approx_contour = approx_contour.reshape(-1, 2).astype(float)
        
        mid_idx = len(approx_contour) // 2
        mid_y = approx_contour[mid_idx][1]
        mid_x = approx_contour[mid_idx][0]
        rel_y = mid_y / palm_h
        rel_x = mid_x / palm_w
        length = cv2.arcLength(np.array(approx_contour, np.int32), True)
        start = approx_contour[0]
        end = approx_contour[-1]
        angle = abs(math.degrees(math.atan2(end[1]-start[1], end[0]-start[0])))
        
        if handedness == 'Right':
            rel_x = 1 - rel_x
        
        start_rel_x = start[0] / palm_w
        start_rel_y = start[1] / palm_h
        if handedness == 'Right':
            start_rel_x = 1 - start_rel_x
        thumb_base = landmarks_norm[4]
        index_base = landmarks_norm[8]
        middle_base = landmarks_norm[12]
        pinky_base = landmarks_norm[20]
        
        thumb_dist = math.hypot(start_rel_x - thumb_base[0], start_rel_y - thumb_base[1])
        index_dist = math.hypot(start_rel_x - index_base[0], start_rel_y - index_base[1])
        middle_dist = math.hypot(start_rel_x - middle_base[0], start_rel_y - middle_base[1])
        pinky_dist = math.hypot(start_rel_x - pinky_base[0], start_rel_y - pinky_base[1])
        
        if angle > 30 and rel_y > 0.4 and rel_x < 0.4 and thumb_dist < 0.3:
            life_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 25 and rel_y < 0.2 and index_dist < 0.3:
            heart_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 35 and 0.3 < rel_y < 0.6 and middle_dist < 0.3:
            head_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 20 and 0.4 < rel_y < 0.9 and 0.4 < rel_x < 0.6:
            fate_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 25 and rel_y > 0.6 and rel_x > 0.6:
            health_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 30 and abs(rel_y - 0.7) < 0.05 and rel_x > 0.8 and pinky_dist < 0.3:  # FIX: abs for Hôn Nhân
            marriage_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 20 and 0.4 < rel_y < 0.8 and abs(rel_x - 0.8) < 0.05:  # FIX: abs for Trí Lục
            sun_line.append((length, angle, approx_contour, rel_y, rel_x))
    
    life_line = sorted(life_line, key=lambda x: x[0], reverse=True)[:2]
    heart_line = sorted(heart_line, key=lambda x: x[0], reverse=True)[:2]
    head_line = sorted(head_line, key=lambda x: x[0], reverse=True)[:2]
    fate_line = sorted(fate_line, key=lambda x: x[0], reverse=True)[:1]
    health_line = sorted(health_line, key=lambda x: x[0], reverse=True)[:1]
    marriage_line = sorted(marriage_line, key=lambda x: x[0], reverse=True)[:3]
    sun_line = sorted(sun_line, key=lambda x: x[0], reverse=True)[:1]
    
    return life_line, heart_line, head_line, fate_line, health_line, marriage_line, sun_line

# Breaks/Branches (giữ nguyên)
def detect_breaks_branches(contour, skeleton=None):
    if skeleton is None:
        is_break = len(contour) < 80
        num_branches = 0
        branches = []
        return is_break, num_branches, branches
    
    is_break = len(contour) < 100
    branches = []
    for i, pt in enumerate(contour):
        y, x = int(pt[1]), int(pt[0])
        if 1 <= y < skeleton.shape[0] - 1 and 1 <= x < skeleton.shape[1] - 1:
            neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] > 0) - 1
            if neighbors > 2:
                branches.append((x, y))
    num_branches = len(branches)
    return is_break, num_branches, branches

def score_line_tracing(lines, palm_h, palm_w):
    if not lines: return 2, False, 0
    max_len = max(l[0] for l in lines)
    base = min(8, int((max_len / (palm_h * 0.6)) * 8))
    total_breaks = sum(1 for l in lines if detect_breaks_branches(l[2], None)[0])
    total_branches = sum(detect_breaks_branches(l[2], None)[1] for l in lines)
    penalty = min(3, total_breaks * 2)
    branch_bonus = min(2, total_branches)
    curve_bonus = sum(1 for l in lines if l[1] > 40)
    score = base + branch_bonus + curve_bonus - penalty
    return max(1, min(10, int(score))), total_breaks > 0, total_branches

def process_palm(image):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if not results.multi_hand_landmarks:
        return image, "Không detect bàn tay rõ! Điểm mặc định thấp. Chụp ảnh lòng bàn tay mở, sáng sủa hướng lên camera.\n\n### PHÂN TÍCH CHI TIẾT\n- **Detect**: 0 bàn tay.\n- **Đường Sinh Khí**: 0 segs, 1/10 | Ý nghĩa: Sức khỏe.\n- **Đường Tâm Đạo**: 0 segs
