import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from disease_info import disease_info
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f4fff4;
}
.stButton>button {
    width: 100%;
    background-color: #2E8B57;
    color: white;
    border-radius: 10px;
    font-size:18px;
    height:55px;
}
.stButton>button:hover{
    background-color:#1f6f43;
}
.result-box{
    padding:15px;
    border-radius:12px;
    background:#e8f5e9;
    border:2px solid #2E8B57;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 Plant Disease Prediction")
st.sidebar.info("""
This application predicts plant leaf diseases using a CNN model trained on the PlantVillage dataset.

Supported Plants:
- 🍅 Tomato
- 🥔 Potato
- 🌶 Pepper
""")

st.title("🌿 Plant Disease Prediction System")
st.write("Upload a plant leaf image and click **Predict Disease**.")

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "plant_disease_model.h5"
FILE_ID = "1EMizC4GFfIT8PWoSriNLWkpvfJQJg-xq"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... Please wait."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        raise
model = load_model()

st.success("✅ Model Loaded Successfully")

# ---------------- CLASS NAMES ----------------
class_names = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

uploaded_file = st.file_uploader(
    "📷 Upload Plant Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):

        img = image.resize((224,224))
        img = np.array(img)
        img = img/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)

        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

        disease = class_names[predicted_index]
        st.markdown("---")

        st.markdown(
            f"""
            <div class="result-box">
            <h2>🌱 Prediction Result</h2>
            <h3>{disease}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### 📊 Confidence")
        st.progress(float(confidence))
        st.success(f"{confidence*100:.2f}%")

        if disease in disease_info:

            info = disease_info[disease]

            st.markdown("## 🦠 Symptoms")
            for item in info["Symptoms"]:
                st.write("•", item)

            st.markdown("## 🧫 Cause")
            st.info(info["Cause"])

            st.markdown("## 🛡 Prevention")
            for item in info["Prevention"]:
                st.write("•", item)
            st.markdown("## 💊 Treatment")
            for item in info["Treatment"]:
                st.write("•", item)

            st.markdown("## 🌱 Plant Care Recommendations")

            if "Water" in info:
                st.write("💧 Water:", info["Water"])

            if "Moisture" in info:
                st.write("💦 Soil Moisture:", info["Moisture"])

            if "Sunlight" in info:
                st.write("☀️ Sunlight:", info["Sunlight"])

            if "Soil" in info:
                st.write("🌱 Soil:", info["Soil"])

            if "Temperature" in info:
                st.write("🌡️ Temperature:", info["Temperature"])

            if "Humidity" in info:
                st.write("💨 Humidity:", info["Humidity"])

            if "Pesticide" in info:
                st.write("🧪 Recommended Pesticide:", info["Pesticide"])

            if "Pesticide Ratio" in info:
                st.write("⚖️ Pesticide Ratio:", info["Pesticide Ratio"])

            if "Composition" in info:
                st.write("🧬 Composition:", info["Composition"])

            if "Spray Interval" in info:
                st.write("📅 Spray Interval:", info["Spray Interval"])

            if "Fertilizer" in info:
                st.write("🌾 Fertilizer:", info["Fertilizer"])

            if "Severity" in info:
                st.write("⚠️ Disease Severity:", info["Severity"])

            if "Recovery Time" in info:
                st.write("⏳ Recovery Time:", info["Recovery Time"])

        else:
            st.warning("Disease information not available.")
        