"""
Streamlit App for Handwritten Digit Recognition
Draw a digit on the canvas and get real-time predictions!
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# Page configuration
st.set_page_config(
    page_title="Digit Recognition - AI Project",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, minimal CSS
st.markdown("""
    <style>
    .main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        padding: 0.6rem 1rem;
        font-size: 15px;
        font-weight: 500;
        border-radius: 8px;
        border: none;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #1B5E20;
    }
    
    h1 {
        color: #1976D2;
        font-weight: 600;
    }
    
    h3 {
        color: #424242;
        font-weight: 500;
    }
    
    .prediction-result {
        background: #E3F2FD;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #1976D2;
        margin: 1rem 0;
    }
    
    .prediction-digit {
        font-size: 80px;
        font-weight: bold;
        color: #1976D2;
        margin: 0.5rem 0;
    }
    
    .confidence-text {
        font-size: 22px;
        color: #424242;
        margin-top: 0.5rem;
    }
    
    .info-box {
        background: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976D2;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = "models/digit_recognition_model.keras"
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found! Please run train_model.py first to train the model.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess image
def preprocess_image(image):
    """Preprocess the drawn image - matches original MS Paint approach"""
    # Convert to grayscale (take first channel)
    if image.shape[2] == 4:
        img_gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours to crop the digit
    contours, _ = cv2.findContours(255 - img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and len(contours) > 0:
        # Get bounding rectangle
        all_points = np.vstack([cnt for cnt in contours])
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add padding
        pad = max(int(0.15 * max(w, h)), 5)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img_gray.shape[1] - x, w + 2*pad)
        h = min(img_gray.shape[0] - y, h + 2*pad)
        
        # Crop to digit
        img_cropped = img_thresh[y:y+h, x:x+w]
    else:
        img_cropped = img_thresh
    
    # Make square
    h, w = img_cropped.shape
    max_dim = max(h, w)
    square = np.ones((max_dim, max_dim), dtype=np.uint8) * 255  # White background
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = img_cropped
    
    # Resize to 28x28
    img_resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # INVERT: black pen on white → white digit on black (like original code)
    img_inverted = np.invert(img_resized)
    
    # Keep as 0-255 values (no normalization)
    img_processed = img_inverted.astype('float32')
    
    # Reshape for model
    img_processed = img_processed.reshape(1, 28, 28, 1)
    
    return img_processed, img_inverted

# Predict digit
def predict_digit(model, image):
    """Make prediction on the preprocessed image"""
    prediction = model.predict(image, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = prediction[0][predicted_digit] * 100
    return predicted_digit, confidence, prediction[0]

# Main app
def main():
    # Simple, clean header
    st.title("🔢 Handwritten Digit Recognition")
    st.markdown("Draw any digit (0-9) and watch the AI recognize it in real-time")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.info("📝 To train the model, run: `python train_model.py`")
        return
    
    # Clean sidebar
    with st.sidebar:
        st.header("⚙️ Drawing Settings")
        stroke_width = st.slider("Pen Size", 10, 35, 25)
        
        st.markdown("---")
        st.header("📋 Instructions")
        st.markdown("""
        1. **Draw** a digit (0-9)
        2. **Use thick strokes**
        3. **Center** your digit
        4. **Clear** to try again
        """)
        
        st.markdown("---")
        st.header("ℹ️ Model Details")
        st.markdown("""
        **Type:** CNN  
        **Dataset:** MNIST  
        **Accuracy:** 98.9%  
        **Input:** 28×28 pixels
        """)
    
    # Main content
    col1, col2 = st.columns([1.3, 1])
    
    # Store prediction result in session state
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    
    with col1:
        st.subheader("🎨 Draw Here")
        
        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=320,
            width=320,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    with col2:
        st.subheader("🤖 Prediction")
        
        if canvas_result.image_data is not None:
            # Check if there's actual drawing on the canvas
            img_data = canvas_result.image_data[:, :, :3]
            # White background is [255, 255, 255], check if any pixels differ significantly
            white_bg = np.full_like(img_data, 255)
            diff = np.abs(img_data.astype(float) - white_bg.astype(float))
            total_diff = np.sum(diff)
            
            # Threshold: need significant pixel changes to consider it a drawing
            MIN_DRAWING_THRESHOLD = 10000
            
            if total_diff > MIN_DRAWING_THRESHOLD:
                # Preprocess image
                processed_img, display_img = preprocess_image(canvas_result.image_data.astype('uint8'))
                
                # Make prediction
                predicted_digit, confidence, all_probs = predict_digit(model, processed_img)
                
                # Store prediction in session state
                st.session_state.last_prediction = predicted_digit
                
                # Clean prediction display
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color: #424242; margin-bottom: 0.5rem;">Predicted Digit</h3>
                    <div class="prediction-digit">{predicted_digit}</div>
                    <div class="confidence-text">Confidence: <strong>{confidence:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show processed image
                st.markdown("---")
                st.markdown("**Model Input (28×28)**")
                st.image(display_img, width=150, clamp=True)
                
                # Show probabilities
                st.markdown("---")
                st.markdown("**Confidence for All Digits**")
                prob_dict = {str(i): float(prob * 100) for i, prob in enumerate(all_probs)}
                st.bar_chart(prob_dict, height=200)
                
            else:
                st.markdown("""
                <div class="info-box">
                    <p style="margin: 0; color: #424242;">👆 <strong>Draw a digit on the canvas</strong> to see the prediction</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Ready to start!")
    
    # Action buttons
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🗑️ Clear Canvas"):
            st.rerun()
    
    with col_btn2:
        if st.button("💾 Save Drawing"):
            if canvas_result.image_data is not None:
                img_data = canvas_result.image_data[:, :, :3]
                white_bg = np.full_like(img_data, 255)
                diff = np.sum(np.abs(img_data.astype(float) - white_bg.astype(float)))
                if diff > 10000 and st.session_state.last_prediction is not None:
                    os.makedirs("saved_digits", exist_ok=True)
                    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                    
                    # Count existing files with same digit
                    existing_files = [f for f in os.listdir('saved_digits') if f.startswith(f'digit_{st.session_state.last_prediction}_')]
                    count = len(existing_files) + 1
                    
                    # Save with predicted digit in filename
                    filename = f"saved_digits/digit_{st.session_state.last_prediction}_{count}.png"
                    img.save(filename)
                    st.success(f"✅ Saved as digit {st.session_state.last_prediction}!")
                else:
                    st.warning("⚠️ Draw and predict a digit first!")
            else:
                st.warning("⚠️ Draw something first!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #757575; padding: 1rem;">
        <p>Built with TensorFlow & Streamlit | 
        <a href="https://github.com/MuhammadMehdiRaza/Hand_Written_Digit_Recogntition" target="_blank" style="color: #1976D2;">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
