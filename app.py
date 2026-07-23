import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Skin Lesion Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Skin Lesion Detection")

st.sidebar.markdown("""
### Project Information

- **Model:** MobileNetV2
- **Dataset:** HAM10000
- **Classes:** 7 Skin Diseases
- **Framework:** TensorFlow + Streamlit

---
Developed for educational purposes.
""")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_skin_model.keras")

model = load_model()

# ---------------- CLASS NAMES ----------------
class_names = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis-like Lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevus (nv)",
    "Vascular Lesions (vasc)"
]

# ---------------- DISEASE INFO ----------------
disease_info = {
    "Actinic Keratoses (akiec)": "A precancerous skin lesion caused by long-term sun exposure.",
    "Basal Cell Carcinoma (bcc)": "The most common type of skin cancer. It usually grows slowly and is highly treatable when detected early.",
    "Benign Keratosis-like Lesions (bkl)": "Non-cancerous skin growths that generally do not require treatment.",
    "Dermatofibroma (df)": "A harmless benign skin nodule.",
    "Melanoma (mel)": "A serious form of skin cancer that requires immediate medical attention.",
    "Melanocytic Nevus (nv)": "A common mole that is usually benign.",
    "Vascular Lesions (vasc)": "Lesions involving blood vessels such as angiomas."
}

# ---------------- TITLE ----------------
st.title("🩺 AI-Powered Skin Lesion Detection")

st.write(
    "Upload a dermatoscopic image and the AI model will classify it into one of the seven skin lesion categories."
)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, width="stretch")

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    probabilities = prediction[0]

    top3 = np.argsort(probabilities)[::-1][:3]

    best = top3[0]

    with col2:

        st.subheader("🔍 Prediction Results")

        st.success(f"### {class_names[best]}")

        st.write(f"**Confidence:** {probabilities[best]*100:.2f}%")

        st.markdown("---")

        st.subheader("📊 Top 3 Predictions")

        for i in top3:
            st.write(f"**{class_names[i]}**")
            st.progress(float(probabilities[i]))
            st.write(f"{probabilities[i]*100:.2f}%")

        st.markdown("---")

        st.subheader("📖 About the Predicted Disease")

        st.info(disease_info[class_names[best]])

# ---------------- DISCLAIMER ----------------
st.markdown("---")

st.warning(
    "⚠️ This application is for educational purposes only and is not a substitute for diagnosis by a qualified dermatologist."
)