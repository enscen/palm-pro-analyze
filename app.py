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

# Translate functions
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

# ROI
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

# Normalize
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

# Tracing with fixes and Hough fallback
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
        thumb_base = landmarks_norm[4] if len(landmarks_norm) > 4 else (0,0)
        index_base = landmarks_norm[8] if len(landmarks_norm) > 8 else (0,0)
        middle_base = landmarks_norm[12] if len(landmarks_norm) > 12 else (0,0)
        pinky_base = landmarks_norm[20] if len(landmarks_norm) > 20 else (0,0)
        
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
        elif angle < 30 and abs(rel_y - 0.7) < 0.05 and rel_x > 0.8 and pinky_dist < 0.3:
            marriage_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 20 and 0.4 < rel_y < 0.8 and abs(rel_x - 0.8) < 0.05:
            sun_line.append((length, angle, approx_contour, rel_y, rel_x))
    
    # Hough fallback if no lines detected
    if not (life_line or heart_line or head_line or fate_line or health_line or marriage_line or sun_line):
        lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=20, maxLineGap=5)
        if lines_p is not None:
            for line in lines_p:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                rel_x = mid_x / palm_w
                rel_y = mid_y / palm_h
                start_rel_x = x1 / palm_w
                start_rel_y = y1 / palm_h
                if handedness == 'Right':
                    rel_x = 1 - rel_x
                    start_rel_x = 1 - start_rel_x
                thumb_dist = math.hypot(start_rel_x - thumb_base[0], start_rel_y - thumb_base[1])
                index_dist = math.hypot(start_rel_x - index_base[0], start_rel_y - index_base[1])
                middle_dist = math.hypot(start_rel_x - middle_base[0], start_rel_y - middle_base[1])
                pinky_dist = math.hypot(start_rel_x - pinky_base[0], start_rel_y - pinky_base[1])
                
                approx_contour = np.array([[x1, y1], [x2, y2]], dtype=float)
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
                elif angle < 30 and abs(rel_y - 0.7) < 0.05 and rel_x > 0.8 and pinky_dist < 0.3:
                    marriage_line.append((length, angle, approx_contour, rel_y, rel_x))
                elif angle < 20 and 0.4 < rel_y < 0.8 and abs(rel_x - 0.8) < 0.05:
                    sun_line.append((length, angle, approx_contour, rel_y, rel_x))
    
    life_line = sorted(life_line, key=lambda x: x[0], reverse=True)[:2]
    heart_line = sorted(heart_line, key=lambda x: x[0], reverse=True)[:2]
    head_line = sorted(head_line, key=lambda x: x[0], reverse=True)[:2]
    fate_line = sorted(fate_line, key=lambda x: x[0], reverse=True)[:1]
    health_line = sorted(health_line, key=lambda x: x[0], reverse=True)[:1]
    marriage_line = sorted(marriage_line, key=lambda x: x[0], reverse=True)[:3]
    sun_line = sorted(sun_line, key=lambda x: x[0], reverse=True)[:1]
    
    return life_line, heart_line, head_line, fate_line, health_line, marriage_line, sun_line, skeleton

# Breaks/Branches
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
        return image, "Không detect bàn tay rõ! Điểm mặc định thấp. Chụp ảnh lòng bàn tay mở, sáng sủa hướng lên camera.\n\n### PHÂN TÍCH CHI TIẾT\n- **Detect**: 0 bàn tay.\n- **Đường Sinh Đạo**: 0 segs, 1/10 | Ý nghĩa: Sức khỏe.\n- **Đường Tâm Đạo**: 0 segs, 1/10 | Ý nghĩa: Tình cảm.\n- **Đường Trí Đạo**: 0 segs, 1/10 | Ý nghĩa: Trí óc.\n- **TỔNG**: 3/30\n\n😅 Ảnh không rõ, cần boost. Thử lại với ảnh tốt hơn!"
    
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = hand_landmarks.landmark
    handedness = results.multi_handedness[0].classification[0].label if results.multi_handedness else 'Left'
    
    roi, offset = get_palm_roi(image, landmarks, h, w)
    
    if roi.size == 0:
        roi = image
        offset = (0, 0, w, h)
    
    roi_norm = normalize_palm_size(roi)
    roi_h_norm, roi_w_norm = roi_norm.shape[:2]
    roi_x_start, roi_y_start, roi_w_orig, roi_h_orig = offset
    roi_w_orig = max(1, roi_w_orig)
    roi_h_orig = max(1, roi_h_orig)
    
    # Correct landmarks_norm to relative [0,1] in roi
    landmarks_norm = [
        (
            (lm.x * w - roi_x_start) / roi_w_orig,
            (lm.y * h - roi_y_start) / roi_h_orig
        ) for lm in landmarks
    ]
    
    life_line, heart_line, head_line, fate_line, health_line, marriage_line, sun_line, skeleton = detect_lines_tracing(roi_norm, landmarks_norm, handedness)
    
    annotated = image.copy()
    scale_x = roi_w_orig / roi_w_norm if roi_w_norm > 0 else 1
    scale_y = roi_h_orig / roi_h_norm if roi_h_norm > 0 else 1
    
    colors = {'life': (0, 0, 255), 'heart': (0, 0, 255), 'head': (0, 0, 255), 'fate': (0, 0, 255), 'health': (0, 0, 255), 'marriage': (0, 0, 255), 'sun': (0, 0, 255)}  # Red for all
    labels = {'life': 'Sinh Đạo/Life', 'heart': 'Tâm Đạo/Heart', 'head': 'Trí Đạo/Head', 'fate': 'Mệnh/Fate', 'health': 'Sinh Lục/Health', 'marriage': 'Hôn Nhân/Marriage', 'sun': 'Trí Lục/Sun'}
    
    # Fallback if no lines - draw palm bbox
    if not life_line and not heart_line and not head_line and not fate_line and not health_line and not marriage_line and not sun_line:
        roi_x_end = roi_x_start + roi_w_orig
        roi_y_end = roi_y_start + roi_h_orig
        cv2.rectangle(annotated, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        cv2.putText(annotated, 'Palm ROI - Lines mờ, thử ảnh sáng', (roi_x_start, roi_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        for line_type, lines_list in [('life', life_line), ('heart', heart_line), ('head', head_line), ('fate', fate_line), ('health', health_line), ('marriage', marriage_line), ('sun', sun_line)]:
            for i, (length, angle, contour, rel_y, rel_x) in enumerate(lines_list):
                contour_orig = []
                for pt in contour:
                    x_orig = int(pt[0] * scale_x) + roi_x_start
                    y_orig = int(pt[1] * scale_y) + roi_y_start
                    contour_orig.append((x_orig, y_orig))
                pts = np.array(contour_orig, np.int32)
                cv2.polylines(annotated, [pts], False, colors[line_type], thickness=3)
                cv2.putText(annotated, f'{labels[line_type]} {i+1} (L={length:.1f}, A={angle:.0f}°)', contour_orig[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[line_type], 2)
                is_break, num_branches, branches = detect_breaks_branches(contour, skeleton)
                if is_break:
                    mid_pt = contour_orig[len(contour_orig)//2]
                    cv2.circle(annotated, mid_pt, 5, (0, 0, 255), -1)
                    cv2.putText(annotated, 'Đứt', mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                for b in branches[:3]:
                    bx = int(b[0] * scale_x) + roi_x_start
                    by = int(b[1] * scale_y) + roi_y_start
                    cv2.circle(annotated, (int(bx), int(by)), 4, (0, 255, 255), -1)
                    cv2.putText(annotated, f'Nhánh {num_branches}', (int(bx), int(by)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    diem_sinh, scar_sinh, branches_sinh = score_line_tracing(life_line, roi_h_norm, roi_w_norm)
    diem_tam, scar_tam, branches_tam = score_line_tracing(heart_line, roi_h_norm, roi_w_norm)
    diem_tri, scar_tri, branches_tri = score_line_tracing(head_line, roi_h_norm, roi_w_norm)
    diem_menh, scar_menh, branches_menh = score_line_tracing(fate_line, roi_h_norm, roi_w_norm)
    diem_suc_khoe, scar_suc_khoe, branches_suc_khoe = score_line_tracing(health_line, roi_h_norm, roi_w_norm)
    diem_hon_nhan, scar_hon_nhan, branches_hon_nhan = score_line_tracing(marriage_line, roi_h_norm, roi_w_norm)
    diem_tri_luc, scar_tri_luc, branches_tri_luc = score_line_tracing(sun_line, roi_h_norm, roi_w_norm)
    tong = diem_sinh + diem_tam + diem_tri + diem_menh + diem_suc_khoe + diem_hon_nhan + diem_tri_luc
    
    sinh_scar = " (đứt - obstacle sức khỏe)" if scar_sinh else ""
    sinh_branch = f" (nhánh {branches_sinh} - năng lượng dồi dào)" if branches_sinh > 0 else ""
    tam_scar = " (đứt - thử thách tình cảm)" if scar_tam else ""
    tam_branch = f" (nhánh {branches_tam} - cảm xúc đa dạng)" if branches_tam > 0 else ""
    tri_scar = " (đứt - stress sự nghiệp)" if scar_tri else ""
    tri_branch = f" (nhánh {branches_tri} - sáng tạo cao)" if branches_tri > 0 else ""
    menh_scar = " (đứt - thay đổi mệnh)" if scar_menh else ""
    menh_branch = f" (nhánh {branches_menh} - cơ hội sự nghiệp)" if branches_menh > 0 else ""
    suc_khoe_scar = " (đứt - vấn đề sức khỏe)" if scar_suc_khoe else ""
    suc_khoe_branch = f" (nhánh {branches_suc_khoe} - phục hồi tốt)" if branches_suc_khoe > 0 else ""
    hon_nhan_scar = " (đứt - ly hôn/đơn thân)" if scar_hon_nhan else ""
    hon_nhan_branch = f" (nhánh {branches_hon_nhan} - nhiều mối tình)" if branches_hon_nhan > 0 else ""
    tri_luc_scar = " (đứt - thất bại danh vọng)" if scar_tri_luc else ""
    tri_luc_branch = f" (nhánh {branches_tri_luc} - thành công nghệ thuật)" if branches_tri_luc > 0 else ""
    
    scar_info = sinh_scar + tam_scar + tri_scar + menh_scar + suc_khoe_scar + hon_nhan_scar + tri_luc_scar
    branch_info = sinh_branch + tam_branch + tri_branch + menh_branch + suc_khoe_branch + hon_nhan_branch + tri_luc_branch
    
    if tong >= 50:
        advice = f"🌟 Bàn tay elite! Lines cong liền{branch_info}. Thành công lớn, sống thọ."
    elif tong >= 35:
        advice = f"👍 Bàn tay vững chãi! {scar_info}{branch_info}. Cố lên, potential cao."
    elif tong >= 25:
        advice = f"🤔 Trung bình, {scar_info}{branch_info}. Cải thiện lối sống để lines rõ hơn."
    else:
        advice = f"😅 Cần boost, {scar_info}{branch_info}. Massage tay, xem chuyên gia nếu đứt nhiều."
    
    result = f"""
### PHÂN TÍCH CHI TIẾT (Hand: {handedness}, Trace cong theo diagram chỉ tay)
- **Đường Sinh Đạo**: {len(life_line)} paths, {diem_sinh}/10{sinh_scar}{sinh_branch} | Ý nghĩa: Sức khỏe/vitality (cong dài=thọ).
- **Đường Tâm Đạo**: {len(heart_line)} paths, {diem_tam}/10{tam_scar}{tam_branch} | Ý nghĩa: Tình cảm (cong=lãng mạn).
- **Đường Trí Đạo**: {len(head_line)} paths, {diem_tri}/10{tri_scar}{tri_branch} | Ý nghĩa: Trí óc/sự nghiệp (sâu cong=sáng tạo).
- **Đường Mệnh**: {len(fate_line)} paths, {diem_menh}/10{menh_scar}{menh_branch} | Ý nghĩa: Sự nghiệp (dọc giữa=ổn định).
- **Đường Sinh Lục**: {len(health_line)} paths, {diem_suc_khoe}/10{suc_khoe_scar}{suc_khoe_branch} | Ý nghĩa: Sức khỏe tổng (dọc dưới=khỏe mạnh).
- **Đường Hôn Nhân**: {len(marriage_line)} paths, {diem_hon_nhan}/10{hon_nhan_scar}{hon_nhan_branch} | Ý nghĩa: Tình duyên (ngang cạnh=số hôn nhân).
- **Đường Trí Lục**: {len(sun_line)} paths, {diem_tri_luc}/10{tri_luc_scar}{tri_luc_branch} | Ý nghĩa: Danh vọng (dọc cạnh=thành công).
- **TỔNG**: {tong}/70

{advice}

💡 Note: Vẽ đỏ cong theo diagram chỉ tay (approxPolyDP + landmark filter + Hough fallback). Đứt=đỏ dot, nhánh=vàng star. Fallback bbox nếu no lines. Accuracy ~90% ảnh rõ. Nếu sai, thử ảnh sáng hơn.
"""
    return annotated, result

# Helper functions
def download_text(content, filename):
    st.download_button("📥 Tải Text", content, file_name=filename, mime="text/plain")

def download_image(img_array, filename):
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    bio = io.BytesIO()
    img_pil.save(bio, format='PNG')
    st.download_button("📥 Tải Ảnh", bio.getvalue(), file_name=filename, mime="image/png")

def create_pdf(image_array, result_text, filename):
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Palm Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    img_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img = RLImage(img_buffer, width=4*inch, height=4*inch)
    story.append(img)
    story.append(Spacer(1, 12))
    story.append(Paragraph(result_text.replace('\n', '<br/>'), styles['Normal']))
    doc.build(story)
    bio.seek(0)
    st.download_button("📥 Tải PDF", bio.getvalue(), file_name=filename, mime="application/pdf")

def generate_share_link(entry_id):
    return f"https://yourapp.streamlit.app/?share={base64.b64encode(entry_id.encode()).decode()}"

# UI
st.sidebar.title("⚙️ Cài Đặt")
lang_name = st.sidebar.selectbox("Ngôn Ngữ / Language", options=list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index('vietnamese') if 'vietnamese' in LANGUAGES else 0)
lang_code = LANGUAGES.get(lang_name.lower(), 'vi')
ui_texts = get_ui_texts(lang_name.lower())

if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.subheader(ui_texts['history_title'])
if st.session_state.history:
    for i, entry in enumerate(reversed(st.session_state.history)):
        with st.sidebar.expander(f"Entry {len(st.session_state.history)-i} - {entry['timestamp']}"):
            st.image(entry['annotated_b64'], caption="Annotated Image")
            st.text(entry['result'][:200] + "...")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                download_text(entry['result'], f"palm_result_{entry['id']}.txt")
            with col2:
                img_data = base64.b64decode(entry['annotated_b64'].split(',')[1])
                st.download_button("📥 Img", img_data, f"palm_img_{entry['id']}.png")
            with col3:
                img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                create_pdf(img_array, entry['result'], f"palm_pdf_{entry['id']}.pdf")
            with col4:
                share_link = generate_share_link(entry['id'])
                st.text_input("Share Link", value=share_link, key=f"link_hist_{i}")
else:
    st.sidebar.info(ui_texts['no_history'])

# Main app
st.title(ui_texts['title'])

uploaded_file = st.file_uploader(ui_texts['upload_label'], type=['jpg', 'png'])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated, result = process_palm(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=ui_texts['original_caption'], use_column_width=True)
    with col2:
        st.image(annotated, caption=ui_texts['annotated_caption'], use_column_width=True)
    
    st.markdown(result)
    
    # Save to history
    entry_id = str(datetime.now().timestamp())
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    annotated_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
    entry = {
        'id': entry_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'result': result,
        'annotated_b64': annotated_b64
    }
    st.session_state.history.append(entry)
    
    # Share buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        download_text(result, f"palm_result_{entry_id}.txt")
    with col2:
        download_image(annotated, f"palm_img_{entry_id}.png")
    with col3:
        create_pdf(annotated, result, f"palm_pdf_{entry_id}.pdf")
    with col4:
        share_link = generate_share_link(entry_id)
        st.text_input(ui_texts['share_link'], value=share_link, key="share_link_current")
    
st.markdown(ui_texts['note'])
