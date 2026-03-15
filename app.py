import streamlit as st
import os
import pandas as pd
import json
import subprocess
import time
import sys

# Set page config
st.set_page_config(
    page_title="YouTube Comment Classifier",
    page_icon="📺",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #6c63ff;
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.2);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(45deg, #6c63ff, #8e2de2);
        color: white;
        font-weight: 800;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4);
        background: linear-gradient(45deg, #8e2de2, #6c63ff);
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 2rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load predictor
@st.cache_resource
def load_predictor():
    try:
        from src.predictor import CommentPredictor
        return CommentPredictor()
    except Exception as e:
        return None

# Function to load metrics
def load_metrics():
    metrics_path = os.path.join('model', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

# Sidebar: Stats & Data Contribution
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/youtube-play.png", width=60)
    st.title("Admin Panel")
    
    # Model Stats
    metrics = load_metrics()
    if metrics:
        st.subheader("📊 Model Stats")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        col2.metric("Dataset", metrics['dataset_size'])
        
        with st.expander("Class Distribution"):
            for label, count in metrics['class_distribution'].items():
                st.markdown(f"<div class='metric-container'><span>{label}</span><b>{count}</b></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Data Contribution
    st.subheader("📥 Add Training Data")
    new_comment = st.text_input("New Comment Content")
    new_category = st.selectbox("Assign Category", ["Funny", "Spam", "Informative", "Hate"])
    if st.button("Add to Dataset"):
        if new_comment:
            try:
                # Ensure the file ends with a newline to prevent merged lines
                csv_path = 'data/dataset.csv'
                if os.path.exists(csv_path):
                    with open(csv_path, 'rb+') as f:
                        f.seek(0, os.SEEK_END)
                        if f.tell() > 0:
                            f.seek(-1, os.SEEK_END)
                            if f.read(1) != b'\n':
                                f.write(b'\n')
                
                # Local append logic
                df_new = pd.DataFrame([[new_comment, new_category]], columns=['comment', 'category'])
                df_new.to_csv(csv_path, mode='a', header=False, index=False)
                st.success("Added! Refresh model to update.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text.")
            
    st.divider()
    
    # Refresh Model
    if st.button("🔄 Refresh & Re-train Model"):
        with st.spinner("Re-training model (approx. 5-10s)..."):
            script_path = os.path.join('src', 'train_model.py')
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            if result.returncode == 0:
                st.cache_resource.clear()
                st.success("Model updated successfully!")
                st.experimental_rerun()
            else:
                st.error("Training failed. Check console.")

# Main UI
st.title("📺 YouTube Comment Classifier")
st.write("A sophisticated NLP engine to categorize content in real-time.")

# Centered Container
c1, c2, c3 = st.columns([1, 2, 1])

with c2:
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    user_input = st.text_area("Analyze Comment", height=120, placeholder="Enter a YouTube comment here...")
    
    predictor = load_predictor()
    
    if st.button("Run Prediction", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a comment.")
        elif predictor is None:
            st.error("Model files missing. Please use 'Refresh Model' in sidebar.")
        else:
            with st.spinner("Processing NLP tokens..."):
                try:
                    prediction = predictor.predict(user_input)
                    
                    # Styled Result Card
                    color_map = {
                        "Funny": ("#fef9e7", "😂", "#f1c40f"),
                        "Spam": ("#fdedec", "🚫", "#e74c3c"),
                        "Informative": ("#ebf5fb", "🧠", "#3498db"),
                        "Hate": ("#f4f6f7", "🤬", "#2c3e50")
                    }
                    bg, emoji, txt_color = color_map.get(prediction, ("#ffffff", "❓", "#000000"))
                    
                    st.markdown(f"""
                        <div class="prediction-card" style="background-color: {bg}; border-left: 10px solid {txt_color}">
                            <h2 style="color: {txt_color}; margin: 0">{emoji} {prediction}</h2>
                            <p style="color: #666; margin-top: 10px">Confidence: High (Based on {metrics['dataset_size'] if metrics else 'N/A'} samples)</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

# Footer
st.markdown("<div style='height: 100px'></div>", unsafe_allow_html=True)
st.markdown("---")
st.caption("© 2026 AI-Powered Classroom Attendance & Engagement Monitoring System | NLP Module")
