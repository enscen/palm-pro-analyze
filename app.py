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
        'title': '🖐️ Palm Pro Analyzer - Chấm Điểm Bàn Tay AI (Tối Ưu)',
        'upload_label': 'Chọn ảnh JPG/PNG',
        'original_caption': 'Ảnh gốc',
        'annotated_caption': 'Ảnh full + Lines overlay đúng vị trí (Xanh=Life, Đỏ=Heart, XanhD=Head)',
        'history_title': 'Lịch Sử Phân Tích',
        'share_text': 'Chia Sẻ Text (.txt)',
        'share_img': 'Chia Sẻ Ảnh (.png)',
        'share_pdf': 'Chia Sẻ PDF',
        'share_link': 'Copy Link Share',
        'no_history': 'Chưa có lịch sử. Upload ảnh để bắt đầu!',
        'detect_error': 'Không detect bàn tay! Chụp rõ lòng bàn tay hướng lên.',
        'note': '💡 Note: Accuracy cao với ảnh sáng. Scar=break >5% palm width (từ palmistry: obstacles). Train ML thêm nếu cần.'
    }
    lang_code = LANGUAGES.get(lang, 'vi')
    translated = {k: translate_text(v, lang_code) for k, v in base_texts.items()}
    return translated

# Fixed ROI: Chỉ crop cho detect, không cho display
def get_palm_roi(image, landmarks, h, w):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    palm_points = points[:5] + points[17:21]  # Wrist, thumb, pinky
    xs = [p[0] for p in palm_points]
    ys = [p[1] for p in palm_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    extend = 0.2
    roi_x_start = max(0, int(min_x - (max_x - min_x) * extend))
    roi_x_end = min(w, int(max_x + (max_x - min_x) * extend))
    roi_y_start = max(0, int(min_y - (max_y - min_y) * extend))
    roi_y_end = min(h, int(max_y + (max_y - min_y) * extend))
    
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    return roi, (roi_x_start, roi_y_start, roi_x_end, roi_y_end)  # Offset full

def normalize_palm_size(roi):
    h, w = roi.shape[:2]
    if h > 0:
        scale = 200 / max(h, w)
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)))
    return roi

def detect_lines_optimized(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closed, 20, 80)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=30)
    
    palm_h, palm_w = roi.shape[:2]
    life_line, heart_line, head_line = [], [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            mid_y = (y1 + y2) / 2
            mid_x = (x1 + x2) / 2
            
            if length > 20:
                rel_y = mid_y / palm_h
                rel_x = mid_x / palm_w
                if angle > 35 and rel_y > 0.4 and rel_x < 0.4:  # Life left-bottom curved
                    life_line.append((length, angle, (x1,y1,x2,y2), rel_y, rel_x))
                elif angle < 20 and rel_y < 0.2:  # Heart top straight
                    heart_line.append((length, angle, (x1,y1,x2,y2), rel_y, rel_x))
                elif angle < 30 and 0.3 < rel_y < 0.6:  # Head middle
                    head_line.append((length, angle, (x1,y1,x2,y2), rel_y, rel_x))
    
    life_line = sorted(life_line, key=lambda x: x[0], reverse=True)[:2]
    heart_line = sorted(heart_line, key=lambda x: x[0], reverse=True)[:2]
    head_line = sorted(head_line, key=lambda x: x[0], reverse=True)[:2]
    
    return life_line, heart_line, head_line

def detect_breaks(line_segments, palm_w):
    if len(line_segments) < 2: return 0, 0
    gaps = []
    for i in range(len(line_segments) - 1):
        seg1 = line_segments[i][2]
        seg2 = line_segments[i+1][2]
        dist = min(math.hypot(seg1[0]-seg2[0], seg1[1]-seg2[1]), math.hypot(seg1[2]-seg2[2], seg1[3]-seg2[3]))
        if dist > palm_w * 0.05:
            gaps.append(dist)
    num_breaks = len(gaps)
    return num_breaks, sum(gaps) / len(gaps) if gaps else 0

def score_line_optimized(lines, palm_h, palm_w):
    if not lines: return 2, False
    max_len = max(l[0] for l in lines)
    base = min(8, int((max_len / (palm_h * 0.6)) * 8))
    straight_bonus = 1 if min(l[1] for l in lines) < 30 else 0
    num_segs = len(lines)
    breaks, avg_gap = detect_breaks(lines, palm_w)
    penalty = min(3, breaks * 1.5 + (avg_gap / palm_w * 2))
    score = base + straight_bonus + min(1, num_segs - 1) - penalty
    return max(1, min(10, int(score))), breaks > 0

def process_palm(image):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if not results.multi_hand_landmarks:
        return image, "Không detect bàn tay rõ! Điểm mặc định thấp. Chụp ảnh lòng bàn tay mở, sáng sủa hướng lên camera.\n\n### PHÂN TÍCH CHI TIẾT\n- **Detect**: 0 bàn tay.\n- **Đường Sinh Khí**: 0 segs, 1/10 | Ý nghĩa: Sức khỏe.\n- **Đường Tâm Đạo**: 0 segs, 1/10 | Ý nghĩa: Tình cảm.\n- **Đường Trí Tuệ**: 0 segs, 1/10 | Ý nghĩa: Trí óc.\n- **TỔNG**: 3/30\n\n😅 Ảnh không rõ, cần boost. Thử lại với ảnh tốt hơn!"
    
    landmarks = results.multi_hand_landmarks[0].landmark
    roi, offset = get_palm_roi(image, landmarks, h, w)  # offset = (x_start, y_start, x_end, y_end)
    
    if roi.size == 0:
        roi = image
        offset = (0, 0, w, h)
    
    roi_norm = normalize_palm_size(roi)
    life, heart, head = detect_lines_optimized(roi_norm)
    
    # FIX: Annotate on FULL image, adjust lines pos with offset
    annotated = image.copy()
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = offset
    roi_h_norm, roi_w_norm = roi_norm.shape[:2]
    roi_h_orig, roi_w_orig = roi.shape[:2]
    scale_x = roi_w_orig / roi_w_norm if roi_w_norm > 0 else 1
    scale_y = roi_h_orig / roi_h_norm if roi_h_norm > 0 else 1
    
    colors = {'life': (0, 255, 0), 'heart': (255, 0, 0), 'head': (0, 0, 255)}
    labels = {'life': 'Sinh Khí', 'heart': 'Tâm Đạo', 'head': 'Trí Tuệ'}
    
    for line_type, lines_list in [('life', life), ('heart', heart), ('head', head)]:
        for i, (length, angle, (x1,y1,x2,y2), rel_y, rel_x) in enumerate(lines_list):
            # Scale back to roi orig
            x1_orig = int(x1 * scale_x) + roi_x_start
            y1_orig = int(y1 * scale_y) + roi_y_start
            x2_orig = int(x2 * scale_x) + roi_x_start
            y2_orig = int(y2 * scale_y) + roi_y_start
            color = colors[line_type]
            thickness = 3 if i==0 else 2
            cv2.line(annotated, (x1_orig, y1_orig), (x2_orig, y2_orig), color, thickness)
            label = f'{labels[line_type]} {i+1} (L={length:.1f}, A={angle:.0f}°)'
            cv2.putText(annotated, label, (x1_orig, y1_orig-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Score (giữ nguyên)
    diem_sinh, scar_sinh = score_line_optimized(life, roi_norm.shape[0], roi_norm.shape[1])
    diem_tam, scar_tam = score_line_optimized(heart, roi_norm.shape[0], roi_norm.shape[1])
    diem_tri, scar_tri = score_line_optimized(head, roi_norm.shape[0], roi_norm.shape[1])
    tong = diem_sinh + diem_tam + diem_tri
    
    scar_info = ""
    if scar_sinh: scar_info += " (có vết sẹo/đứt - obstacle tạm thời, tập trung sức khỏe)"
    if scar_tam: scar_info += " (có vết sẹo - thử thách tình cảm, cần kiên nhẫn)"
    if scar_tri: scar_info += " (có vết sẹo - stress sự nghiệp, nghỉ ngơi đi)"
    
    if tong >= 25:
        advice = f"🌟 Bàn tay elite! Lines rõ dài, {scar_info}. Thành công lớn, sống thọ."
    elif tong >= 18:
        advice = f"👍 Bàn tay vững chãi! {scar_info}. Cố lên, potential cao."
    elif tong >= 12:
        advice = f"🤔 Trung bình, {scar_info}. Cải thiện lối sống để lines rõ hơn."
    else:
        advice = f"😅 Cần boost, {scar_info}. Massage tay, xem chuyên gia nếu scar nhiều."
    
    result = f"""
### PHÂN TÍCH CHI TIẾT (ROI full bàn tay: {roi.shape[:2]} - Detect: {len(results.multi_hand_landmarks)} tay)
- **Đường Sinh Khí**: {len(life)} segs, {diem_sinh}/10{scar_info if scar_sinh else ''} | Ý nghĩa: Sức khỏe/vitality (dài=thọ).
- **Đường Tâm Đạo**: {len(heart)} segs, {diem_tam}/10{scar_info if scar_tam else ''} | Ý nghĩa: Tình cảm (cong=lãng mạn).
- **Đường Trí Tuệ**: {len(head)} segs, {diem_tri}/10{scar_info if scar_tri else ''} | Ý nghĩa: Trí óc/sự nghiệp (sâu=sáng tạo).
- **TỔNG**: {tong}/30

{advice}

💡 Note: Lines vẽ đúng vị trí bàn tay. Nếu lệch, thử ảnh rõ hơn. Accuracy cao với ảnh sáng. Scar=break >5% palm width (từ palmistry: obstacles). Train ML thêm nếu cần.
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
