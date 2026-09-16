import streamlit as st
import cv2
import easyocr
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# 1. Cấu hình trang web (Dùng chế độ hiển thị rộng "wide" để chia cột đẹp hơn)
st.set_page_config(page_title="Text Extraction System", layout="wide")

# 2. Tùy biến giao diện bằng CSS
st.markdown("""
    <style>
    /* Nền tối sâu cho toàn bộ trang web */
    .main {
        background-color: #0f1116;
    }
    
    /* Căn chỉnh tiêu đề chính chính giữa, hiệu ứng sáng Cyan */
    h1 {
        color: #00ffcc !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        letter-spacing: 2px;
        text-shadow: 0px 0px 12px rgba(0, 255, 204, 0.4);
        margin-bottom: 2rem !important;
    }
    
    /* Thiết kế nút bấm "Detect Text" chính dạng Gradient */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00ffcc 0%, #0099ff 100%) !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important; /* Nút kéo dài hết khung bên trái */
        transition: all 0.3s ease;
        box-shadow: 0px 4px 15px rgba(0, 255, 204, 0.3);
    }
    
    /* Hiệu ứng di chuột cho nút chính */
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 20px rgba(0, 255, 204, 0.5);
    }

    /* Thiết kế nút tải file CSV */
    div.stDownloadButton > button {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #4b5563 !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    div.stDownloadButton > button:hover {
        border-color: #00ffcc !important;
        color: #00ffcc !important;
    }
    
    /* Khung tải ảnh kéo thả */
    section[data-testid="stFileUploadDropzone"] {
        background-color: #1a1f2c !important;
        border: 2px dashed #00ffcc !important;
        border-radius: 12px !important;
    }
    
    /* Thêm đường bo góc cho bảng dữ liệu */
    div[data-testid="stDataFrame"] {
        border: 1px solid #2e3748;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


# 3. Logic tải mô hình nhận diện (Giữ nguyên)
@st.cache_resource
def load_models():
    model = YOLO("runs/detect/train/weights/best.pt")
    reader = easyocr.Reader(['vi', 'en'])
    return model, reader

try:
    model, reader = load_models()
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}. Vui lòng kiểm tra lại đường dẫn file weights!")

# 4. Giao diện tiêu đề
st.title("TEXT EXTRACTION SYSTEM")

# 5. Khởi tạo cấu trúc 2 cột với tỷ lệ kích thước 4.5 : 5.5
col_left, col_right = st.columns([4.5, 5.5], gap="large")

# --- CỘT BÊN TRÁI: KHU VỰC ĐIỀU KHIỂN & TẢI ẢNH ---
with col_left:
    st.subheader("📁 Input Source")
    uploaded_file = st.file_uploader(
        label="Chọn hoặc kéo thả hình ảnh vào đây", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh gốc đã tải lên", use_container_width=True)
        
    # Đặt nút bấm hành động ngay bên dưới ảnh đầu vào
    btn_detect = st.button("Detect Text", type="primary")


# --- CỘT BÊN PHẢI: KHU VỰC HIỂN THỊ KẾT QUẢ ---
with col_right:
    st.subheader("📊 Extraction Results")
    
    if btn_detect:
        if uploaded_file is not None:
            with st.spinner("Hệ thống đang trích xuất toàn bộ văn bản, vui lòng đợi..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                output_image = image.copy()
                _ = model(image, conf=0.1)
                detected_data = []
                          
                text_results = reader.readtext(
                    image, 
                    width_ths=0.7,      
                    add_margin=0.15,    
                    low_text=0.3       
                )
                
                for text_result in text_results:
                    bbox, text, confidence = text_result
                    detected_data.append({
                        "Text": text,
                        "Confidence": confidence
                    })
                    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Hiển thị ảnh sau xử lý
                st.markdown("#### Processed Image:")
                rgb_output = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                st.image(rgb_output, caption="Ảnh đã được khoanh vùng văn bản", use_container_width=True) 
                
                # Hiển thị bảng dữ liệu text trích xuất được
                st.markdown("#### Extracted Text & Data Summary:")
                if len(detected_data) > 0:
                    df = pd.DataFrame(detected_data)
                    st.dataframe(df, use_container_width=True, height=200) # Giới hạn chiều cao bảng để tránh kéo quá dài
                    
                    # Hiển thị độ chính xác và nút tải file cạnh nhau cho gọn
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        avg_conf = df['Confidence'].mean()
                        st.metric(label="🎯 Độ chính xác trung bình", value=f"{avg_conf:.2f}")
                    with meta_col2:
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.write("") # Tạo khoảng trống căn lề trên cho nút
                        st.download_button(
                            label="📥 Tải file CSV",
                            data=csv_data,
                            file_name="ocr_comprehensive_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Không tìm thấy đoạn văn bản nào trong hình ảnh!")
        else:
            st.warning("Vui lòng tải một hình ảnh lên trước ở cột bên trái!")
    else:
        # Trạng thái chờ khi người dùng chưa bấm nút Detect
        st.info("Nhấn nút 'Detect Text' sau khi tải ảnh để xem kết quả trích xuất tại đây.")
