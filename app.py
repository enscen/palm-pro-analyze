import streamlit as st
st.set_page_config(page_title="Palm Analyzer", layout="wide")  # KEY FIX: Move to TOP, right after imports!

try:
    import cv2  # Giữ nguyên, headless sẽ handle
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
    LANGUAGES = {'english': 'en', 'vietnamese': 'vi'}  # Fallback dict
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
    return mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8), mp_drawing

hands, mp_drawing = load_mediapipe()

# Translate functions
def translate_text(text, target_lang='vi'):
    """Translate text to target lang (dynamic, all langs supported)"""
    try:
        if not translator or target_lang == 'en': return text
        lang_code = LANGUAGES.get(target_lang, 'vi')
        result = translator.translate(text, dest=lang_code)
        return result.text
    except Exception as e:
        st.warning(f"Translate error ({e}) - fallback to English.")
        return text

def get_ui_texts(lang):
    """UI keys translated"""
    base_texts = {
        'title': '🖐️ Palm Pro Analyzer - Chấm Điểm Bàn Tay AI (Tối Ưu)',
        'upload_label': 'Chọn ảnh JPG/PNG',
        'original_caption': 'Ảnh gốc',
        'annotated_caption': 'Ảnh + Lines detect (Xanh=Life, Đỏ=Heart, XanhD=Head)',
        'history_title': 'Lịch Sử Phân Tích',
        'share_text': 'Chia Sẻ Text (.txt)',
        'share_img': 'Chia Sẻ Ảnh (.png)',
        'share_pdf': 'Chia Sẻ PDF',
        'share_link': 'Copy Link Share',
        'no_history': 'Chưa có lịch sử. Upload ảnh để bắt đầu!',
        'detect_error': 'Không detect bàn tay! Chụp rõ lòng bàn tay hướng lên.',
        'note': '💡 Note: Accuracy cao với ảnh sáng. Scar=break >5% palm width (từ palmistry: obstacles). Train ML thêm nếu cần.'
    }
    lang_code = LANGUAGES.get(lang, 'vi')  # Map name to code, default vi
    translated = {k: translate_text(v, lang_code) for k, v in base_texts.items()}
    return translated

# Các functions detect/score giữ nguyên
def normalize_palm_size(roi):
    h, w = roi.shape[:2]
    if h > 0:
        scale = 200 / h
        roi = cv2.resize(roi, (int(w * scale), 200))
    return roi

def detect_lines_optimized(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closed, 30, 100)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=15)
    
    palm_h, palm_w = roi.shape[:2]
    life_line, heart_line, head_line = [], [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            mid_y = (y1 + y2) / 2
            mid_x = (x1 + x2) / 2
            
            if length > 25:
                if angle > 40 and mid_y > palm_h * 0.5 and mid_x < palm_w * 0.3:
                    life_line.append((length, angle, line[0], mid_y / palm_h))
                elif angle < 25 and mid_y < palm_h * 0.25:
                    heart_line.append((length, angle, line[0], mid_y / palm_h))
                elif angle < 35 and 0.25 < mid_y / palm_h < 0.55:
                    head_line.append((length, angle, line[0], mid_y / palm_h))
    
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
    base = min(8, int((max_len / (palm_h * 0.7)) * 8))
    straight_bonus = 1 if min(l[1] for l in lines) < 30 else 0
    num_segs = len(lines)
    breaks, avg_gap = detect_breaks(lines, palm_w)
    penalty = min(3, breaks * 1.5 + (avg_gap / palm_w * 2))
    score = base + straight_bonus + min(1, num_segs - 1) - penalty
    return max(1, min(10, int(score))), breaks > 0

def process_palm(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if not results.multi_hand_landmarks:
        return None, "Không detect bàn tay! Chụp rõ lòng bàn tay hướng lên."
    
    landmarks = results.multi_hand_landmarks[0].landmark
    h, w = image.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    
    wrist = points[0]
    knuckles_y = max(p[1] for p in points[5:18:4])
    thumb_base = points[4]
    roi_y_start = min(wrist[1], knuckles_y)
    roi_h = abs(wrist[1] - knuckles_y) + 80
    roi = image[roi_y_start:roi_y_start + roi_h, 0:w]
    if roi.size == 0: roi = image
    
    roi_norm = normalize_palm_size(roi)
    life, heart, head = detect_lines_optimized(roi_norm)
    
    annotated = roi.copy()
    colors = {'life': (0, 255, 0), 'heart': (255, 0, 0), 'head': (0, 0, 255)}
    labels = {'life': 'Sinh Khí', 'heart': 'Tâm Đạo', 'head': 'Trí Tuệ'}
    
    for line_type, lines_list in [('life', life), ('heart', heart), ('head', head)]:
        if lines_list:
            strongest = max(lines_list, key=lambda x: x[0])
            x1, y1, x2, y2 = strongest[2]
            cv2.line(annotated, (x1, y1), (x2, y2), colors[line_type], 3)
            cv2.putText(annotated, f'{labels[line_type]} (L={strongest[0]:.1f})', (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[line_type], 2)
    
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
### PHÂN TÍCH CHI TIẾT
- **Detect**: 1 bàn tay, Palm normalized 200px.
- **Đường Sinh Khí**: {len(life)} segs, {diem_sinh}/10{scar_info if scar_sinh else ''} | Ý nghĩa: Sức khỏe/vitality (dài=thọ).
- **Đường Tâm Đạo**: {len(heart)} segs, {diem_tam}/10{scar_info if scar_tam else ''} | Ý nghĩa: Tình cảm (cong=lãng mạn).
- **Đường Trí Tuệ**: {len(head)} segs, {diem_tri}/10{scar_info if scar_tri else ''} | Ý nghĩa: Trí óc/sự nghiệp (sâu=sáng tạo).
- **TỔNG**: {tong}/30

{advice}

💡 Note: Accuracy cao với ảnh sáng. Scar=break >5% palm width (từ palmistry: obstacles). Train ML thêm nếu cần.
"""
    return annotated, result

# Helper: Download functions (giữ nguyên)
def download_text(content, filename):
    st.download_button("📥 Tải Text", content, file_name=filename, mime="text/plain")

def download_image(img_array, filename):
    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    bio = io.BytesIO()
    img_pil.save(bio, format='PNG')
    st.download_button("📥 Tải Ảnh", bio.getvalue(), file_name=filename, mime="image/png")

def create_pdf(image_array, result_text, filename):
    """Tạo PDF với reportlab"""
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Palm Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Image
    img_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img = RLImage(img_buffer, width=4*inch, height=4*inch)
    story.append(img)
    story.append(Spacer(1, 12))
    
    # Text
    story.append(Paragraph(result_text.replace('\n', '<br/>'), styles['Normal']))
    
    doc.build(story)
    bio.seek(0)
    st.download_button("📥 Tải PDF", bio.getvalue(), file_name=filename, mime="application/pdf")

def generate_share_link(entry_id):
    """Simple base64 link for share (or use st.secrets for full URL)"""
    return f"https://yourapp.streamlit.app/?share={base64.b64encode(entry_id.encode()).decode()}"

# Sidebar: Lang + History
st.sidebar.title("⚙️ Cài Đặt")
lang_name = st.sidebar.selectbox("Ngôn Ngữ / Language", options=list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index('vietnamese') if 'vietnamese' in LANGUAGES else 0)
lang_code = LANGUAGES.get(lang_name.lower(), 'vi')
ui_texts = get_ui_texts(lang_name.lower())

if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.subheader(ui_texts['history_title'])
if st.session_state.history:
    for i, entry in enumerate(reversed(st.session_state.history)):  # Latest first
        with st.sidebar.expander(f"Entry {len(st.session_state.history)-i} - {entry['timestamp']}"):
            st.image(entry['annotated_b64'], caption="Annotated Image")
            st.text(entry['result'][:200] + "...")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                download_text(entry['result'], f"palm_result_{entry['id']}.txt")
            with col2:
                # Decode b64 for img download
                img_data = base64.b64decode(entry['annotated_b64'].split(',')[1])
                st.download_button("📥 Img", img_data, f"palm_img_{entry['id']}.png", "image/png")
            with col3:
                # PDF needs img array - recreate from b64
                img_array = cv2.imdecode(np.frombuffer(base64.b64decode(entry['annotated_b64'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
                create_pdf(img_array, entry['result'], f"palm_pdf_{entry['id']}.pdf")
            with col4:
                share_link = generate_share_link(entry['id'])
                st.code(share_link)  # Copyable link
else:
    st.sidebar.info(ui_texts['no_history'])

# Main UI
st.title(translate_text(ui_texts['title'], lang_code))

uploaded_file = st.file_uploader(translate_text(ui_texts['upload_label'], lang_code), type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=translate_text(ui_texts['original_caption'], lang_code), use_column_width=True)
    
    # Process
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated, raw_result = process_palm(image_cv)
    
    if annotated is not None:
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)
        
        # Translate result
        translated_result = translate_text(raw_result, lang_code)
        
        st.image(annotated_pil, caption=translate_text(ui_texts['annotated_caption'], lang_code), use_column_width=True)
        st.markdown(translated_result)
        st.markdown(translate_text(ui_texts['note'], lang_code))
        
        # Save to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry_id = base64.b64encode(os.urandom(8)).decode()  # Unique ID
        _, annotated_b64 = cv2.imencode('.png', annotated_rgb)
        b64_str = "data:image/png;base64," + base64.b64encode(annotated_b64).decode()
        
        st.session_state.history.append({
            'id': entry_id,
            'timestamp': timestamp,
            'result': translated_result,  # Save translated
            'annotated_b64': b64_str
        })
        
        # Quick share buttons (current result)
        col1, col2, col3 = st.columns(3)
        with col1:
            download_text(translated_result, f"palm_result_{entry_id}.txt")
        with col2:
            bio = io.BytesIO()
            annotated_pil.save(bio, format='PNG')
            st.download_button("📥 Img", bio.getvalue(), f"palm_img_{entry_id}.png", "image/png")
        with col3:
            create_pdf(annotated_rgb, translated_result, f"palm_pdf_{entry_id}.pdf")
        
        st.info(f"Đã lưu vào lịch sử! Link share: {generate_share_link(entry_id)}")
    else:
        st.error(translate_text(raw_result, lang_code))

# Footer
st.markdown("---")
st.info("App open-source. Deploy trên Streamlit Cloud để share dễ dàng!")
