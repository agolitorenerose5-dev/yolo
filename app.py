import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# 1. Page Configuration
st.set_page_config(
    page_title="Fruit Detection App",
    page_icon="🍎",
    layout="centered"
)

# 2. Load the Custom YOLO Model
# Using st.cache_resource so the model only loads into memory once
@st.cache_resource
def load_model():
    # Update this path if your 'best.pt' file moves
    model_path = r"best.pt"
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model. Check your path in the code!\nDetails: {e}")
    st.stop()

# 3. App Header UI
st.title("🍎 Fruit Detection App")
st.write("Upload an image to detect fruits using your trained YOLOv8 model.")

# 4. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    
    # Create two columns for Before vs After display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with st.spinner("Detecting fruits..."):
        # 5. Run Inference
        # We pass the PIL image directly. conf=0.25 is standard, feel free to tweak.
        results = model(image, conf=0.25)
        
        # Extract the first result object
        result = results[0]
        
        # Plot returns a numpy array (BGR format) representing the annotated image
        annotated_image_bgr = result.plot()
        
        # Convert BGR (OpenCV format) to RGB (PIL/Streamlit format)
        annotated_image_rgb = annotated_image_bgr[:, :, ::-1]
        
    with col2:
        st.subheader("Object Detection")
        st.image(annotated_image_rgb, use_container_width=True)
        
    # 6. Parse and Display Metrics/Counts
    st.markdown("---")
    st.subheader("📊 Detection Summary")
    
    boxes = result.boxes
    if len(boxes) == 0:
        st.info("No fruits detected. Try lowering the confidence threshold or using a clearer image.")
    else:
        # Build a tally dictionary for detected classes
        detection_counts = {}
        
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
        # Display the counts cleanly
        st.write(f"**Total objects found:** {len(boxes)}")
        
        # Display as bullet points with emojis if matching common items
        for fruit, count in detection_counts.items():
            st.write(f"- **{fruit.capitalize()}**: {count}")
