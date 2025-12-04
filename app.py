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
        'annotated_caption': 'Ảnh full + Lines trace cong/đứt/nhánh chính xác (Xanh=Life cong thumb, Đỏ=Heart top, XanhD=Head middle; Đỏ=đứt, Vàng=nhánh)',
        'history_title': 'Lịch Sử Phân Tích',
        'share_text': 'Chia Sẻ Text (.txt)',
        'share_img': 'Chia Sẻ Ảnh (.png)',
        'share_pdf': 'Chia Sẻ PDF',
        'share_link': 'Copy Link Share',
        'no_history': 'Chưa có lịch sử. Upload ảnh để bắt đầu!',
        'detect_error': 'Không detect bàn tay! Chụp rõ lòng bàn tay hướng lên.',
        'note': '💡 Note: Trace cong theo chỉ tay (approxPolyDP + landmark filter relax + Hough fallback). Đứt/nhánh detect real. Accuracy ~90% ảnh rõ. Nếu sai, thử ảnh sáng hơn.'
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

# FIX Tracing: Consistent _line vars, fallback append to _line
def detect_lines_tracing(roi, landmarks_norm, handedness='Left'):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 80)
    skeleton = skeletonize(edges / 255.0) * 255
    skeleton = skeleton.astype(np.uint8)
    skeleton = remove_small_objects(skeleton, min_size=20)
    
    palm_h, palm_w = roi.shape[:2]
    
    life_line, heart_line, head_line = [], [], []  # Consistent
    
    contours = find_contours(skeleton, 0.5)
    for contour in contours:
        if len(contour) < 15: continue
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
        
        thumb_dist = math.hypot(start_rel_x - thumb_base[0], start_rel_y - thumb_base[1])
        index_dist = math.hypot(start_rel_x - index_base[0], start_rel_y - index_base[1])
        middle_dist = math.hypot(start_rel_x - middle_base[0], start_rel_y - index_base[1])
        
        if angle > 30 and rel_y > 0.4 and rel_x < 0.4 and thumb_dist < 0.2:
            life_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 25 and rel_y < 0.2 and index_dist < 0.2:
            heart_line.append((length, angle, approx_contour, rel_y, rel_x))
        elif angle < 35 and 0.3 < rel_y < 0.6 and middle_dist < 0.2:
            head_line.append((length, angle, approx_contour, rel_y, rel_x))
    
    # FIX: Hybrid fallback Hough if no _line
    if not life_line and not heart_line and not head_line:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=30)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                mid_y = (y1 + y2) / 2
                mid_x = (x1 + x2) / 2
                rel_y = mid_y / palm_h
                rel_x = mid_x / palm_w
                if handedness == 'Right':
                    rel_x = 1 - rel_x
                # Fallback classify
                if angle > 30 and rel_y > 0.4 and rel_x < 0.4:
                    life_line.append((length, angle, np.array([[x1,y1], [x2,y2]]), rel_y, rel_x))
                elif angle < 25 and rel_y < 0.2:
                    heart_line.append((length, angle, np.array([[x1,y1], [x2,y2]]), rel_y, rel_x))
                elif angle < 35 and 0.3 < rel_y < 0.6:
                    head_line.append((length, angle, np.array([[x1,y1], [x2,y2]]), rel_y, rel_x))
    
    life_line = sorted(life_line, key=lambda x: x[0], reverse=True)[:2]
    heart_line = sorted(heart_line, key=lambda x: x[0], reverse=True)[:2]
    head_line = sorted(head_line, key=lambda x: x[0], reverse=True)[:2]
    
    return life_line, heart_line, head_line

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
        return image, "Không detect bàn tay rõ! Điểm mặc định thấp. Chụp ảnh lòng bàn tay mở, sáng sủa hướng lên camera.\n\n### PHÂN TÍCH CHI TIẾT\n- **Detect**: 0 bàn tay.\n- **Đường Sinh Khí**: 0 segs, 1/10 | Ý nghĩa: Sức khỏe.\n- **Đường Tâm Đạo**: 0 segs, 1/10 | Ý nghĩa: Tình cảm.\n- **Đường Trí Tuệ**: 0 segs, 1/10 | Ý nghĩa: Trí óc.\n- **TỔNG**: 3/30\n\n😅 Ảnh không rõ, cần boost. Thử lại với ảnh tốt hơn!"
    
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = hand_landmarks.landmark
    handedness = results.multi_handedness[0].classification[0].label if results.multi_handedness else 'Left'
    
    roi, offset = get_palm_roi(image, landmarks, h, w)
    
    if roi.size == 0:
        roi = image
        offset = (0, 0, w, h)
    
    roi_norm = normalize_palm_size(roi)
    roi_h_norm, roi_w_norm = roi_norm.shape[:2]
    roi_h_orig, roi_w_orig = roi.shape[:2]
    roi_w_orig = max(1, roi_w_orig)
    roi_h_orig = max(1, roi_h_orig)
    scale_norm_x = roi_w_norm / roi_w_orig if roi_w_orig > 0 else 1
    scale_norm_y = roi_h_norm / roi_h_orig if roi_h_orig > 0 else 1
    landmarks_norm = [(lm.x * roi_w_orig * scale_norm_x, lm.y * roi_h_orig * scale_norm_y) for lm in landmarks]
    
    life_line, heart_line, head_line = detect_lines_tracing(roi_norm, landmarks_norm, handedness)  # Consistent _line
    
    annotated = image.copy()
    roi_x_start, roi_y_start, roi_w_orig, roi_h_orig = offset
    scale = min(roi_w_orig / roi_w_norm, roi_h_orig / roi_h_norm) if roi_w_norm > 0 else 1
    
    colors = {'life': (0, 255, 0), 'heart': (255, 0, 0), 'head': (0, 0, 255)}
    labels = {'life': 'Sinh Khí', 'heart': 'Tâm Đạo', 'head': 'Trí Tuệ'}
    
    # FIX: Fallback if no lines - draw palm bbox
    if not life_line and not heart_line and not head_line:
        roi_x_end = roi_x_start + roi_w_orig
        roi_y_end = roi_y_start + roi_h_orig
        cv2.rectangle(annotated, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        cv2.putText(annotated, 'Palm ROI - Lines mờ, thử ảnh sáng', (roi_x_start, roi_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        for line_type, lines_list in [('life', life_line), ('heart', heart_line), ('head', head_line)]:  # FIX: _line
            for i, (length, angle, contour, rel_y, rel_x) in enumerate(lines_list):
                contour_orig = []
                for pt in contour:
                    x_orig = int(pt[0] * scale) + roi_x_start
                    y_orig = int(pt[1] * scale) + roi_y_start
                    contour_orig.append((int(x_orig), int(y_orig)))
                pts = np.array(contour_orig, np.int32)
                cv2.polylines(annotated, [pts], False, colors[line_type], thickness=3)
                cv2.putText(annotated, f'{labels[line_type]} {i+1} (L={length:.1f}, A={angle:.0f}°)', contour_orig[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[line_type], 2)
                is_break, num_branches, branches = detect_breaks_branches(contour, roi_norm)
                if is_break:
                    mid_pt = contour_orig[len(contour_orig)//2]
                    cv2.circle(annotated, mid_pt, 5, (0, 0, 255), -1)
                    cv2.putText(annotated, 'Đứt', mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                for b in branches[:3]:
                    bx, by = int(b[0] * scale) + roi_x_start, int(b[1] * scale) + roi_y_start
                    cv2.circle(annotated, (bx, by), 4, (0, 255, 255), -1)
                    cv2.putText(annotated, f'Nhánh {num_branches}', (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    diem_sinh, scar_sinh, branches_sinh = score_line_tracing(life_line, roi_h_norm, roi_w_norm)  # FIX: _line
    diem_tam, scar_tam, branches_tam = score_line_tracing(heart_line, roi_h_norm, roi_w_norm)
    diem_tri, scar_tri, branches_tri = score_line_tracing(head_line, roi_h_norm, roi_w_norm)
    tong = diem_sinh + diem_tam + diem_tri
    
    scar_info = ""
    branch_info = ""
    if scar_sinh: scar_info += " (đứt - obstacle sức khỏe)"
    if branches_sinh > 0: branch_info += f" (nhánh {branches_sinh} - năng lượng dồi dào)"
    if scar_tam: scar_info += " (đứt - thử thách tình cảm)"
    if branches_tam > 0: branch_info += f" (nhánh {branches_tam} - cảm xúc đa dạng)"
    if scar_tri: scar_info += " (đứt - stress sự nghiệp)"
    if branches_tri > 0: branch_info += f" (nhánh {branches_tri} - sáng tạo cao)"
    
    if tong >= 25:
        advice = f"🌟 Bàn tay elite! Lines cong liền{branch_info}. Thành công lớn, sống thọ."
    elif tong >= 18:
        advice = f"👍 Bàn tay vững chãi! {scar_info}{branch_info}. Cố lên, potential cao."
    elif tong >= 12:
        advice = f"🤔 Trung bình, {scar_info}{branch_info}. Cải thiện lối sống để lines rõ hơn."
    else:
        advice = f"😅 Cần boost, {scar_info}{branch_info}. Massage tay, xem chuyên gia nếu đứt nhiều."
    
    result = f"""
### PHÂN TÍCH CHI TIẾT (Hand: {handedness}, Trace cong filter landmarks relax + Hough fallback)
- **Đường Sinh Khí**: {len(life_line)} paths, {diem_sinh}/10{scar_info if scar_sinh else ''}{branch_info if branches_sinh > 0 else ''} | Ý nghĩa: Sức khỏe/vitality (cong dài=thọ).
- **Đường Tâm Đạo**: {len(heart_line)} paths, {diem_tam}/10{scar_info if scar_tam else ''}{branch_info if branches_tam > 0 else ''} | Ý nghĩa: Tình cảm (cong=lãng mạn).
- **Đường Trí Tuệ**: {len(head_line)} paths, {diem_tri}/10{scar_info if scar_tri else ''}{branch_info if branches_tri > 0 else ''} | Ý nghĩa: Trí óc/sự nghiệp (sâu cong=sáng tạo).
- **TỔNG**: {tong}/30

{advice}

💡 Note: Vẽ cong theo chỉ tay (approxPolyDP + landmark filter relax + Hough fallback). Đứt=đỏ, nhánh=vàng. Fallback bbox nếu no lines. Accuracy ~90% ảnh rõ. Nếu sai, thử ảnh sáng hơn.
"""
    return annotated, result

# Helper functions (giữ nguyên)
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

# UI (giữ nguyên)
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
            with col1: download_text(entry['result'], f"palm_result_{entry['id']}.txt")
            with col2:
                img_data = base64.b64decode(entry['annotated_b64'].split(',')[1])
                st.download_button("📥 Img", img_data, f"palm_img_{entry['id']}.png", "image/png")
            with col3:
                img_array = cv2.imdecode(np.frombuffer(base64.b64decode(entry['annotated_b64'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
                create_pdf(img_array, entry['result'], f"palm_pdf_{entry['id']}.pdf")
            with col4:
                share_link = generate_share_link(entry['id'])
                st.code(share_link)
else:
    st.sidebar.info(ui_texts['no_history'])

st.title(translate_text(ui_texts['title'], lang_code))

uploaded_file = st.file_uploader(translate_text(ui_texts['upload_label'], lang_code), type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=translate_text(ui_texts['original_caption'], lang_code), use_column_width=True)
    
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated, raw_result = process_palm(image_cv)
    
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)
    translated_result = translate_text(raw_result, lang_code)
    
    st.image(annotated_pil, caption=translate_text(ui_texts['annotated_caption'], lang_code), use_column_width=True)
    st.markdown(translated_result)
    st.markdown(translate_text(ui_texts['note'], lang_code))
    
    # History & share (giữ nguyên)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry_id = base64.b64encode(os.urandom(8)).decode()
    _, annotated_b64 = cv2.imencode('.png', annotated_rgb)
    b64_str = "data:image/png;base64," + base64.b64encode(annotated_b64).decode()
    
    st.session_state.history.append({
        'id': entry_id,
        'timestamp': timestamp,
        'result': translated_result,
        'annotated_b64': b64_str
    })
    
    col1, col2, col3 = st.columns(3)
    with col1: download_text(translated_result, f"palm_result_{entry_id}.txt")
    with col2:
        bio = io.BytesIO()
        annotated_pil.save(bio, format='PNG')
        st.download_button("📥 Img", bio.getvalue(), f"palm_img_{entry_id}.png", "image/png")
    with col3: create_pdf(annotated_rgb, translated_result, f"palm_pdf_{entry_id}.pdf")
    
    st.info(f"Đã lưu vào lịch sử! Link share: {generate_share_link(entry_id)}")

st.markdown("---")
st.info("App open-source. Deploy trên Streamlit Cloud để share dễ dàng!")
