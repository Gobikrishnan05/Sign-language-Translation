import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from rembg import remove
from PIL import Image
from gtts import gTTS
import json
import os
import re
import traceback

# ---------------- CONFIG ----------------
MODEL_PATH = "Mobile_98.keras"
CLASS_LABELS_PATH = "class_labels.json"
PLACEHOLDER_IMAGE = "not_sign.png"  # safer relative path
# ----------------------------------------

st.set_page_config(page_title="Tamil ↔ Malayalam Sign Recognition", layout="wide")

# ---------------- LOAD MODEL & LABELS ----------------
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)

    with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # If labels saved as list -> convert to index dictionary
    if isinstance(labels, list):
        labels = {i: v.strip() for i, v in enumerate(labels)}
    else:
        labels = {int(k): v.strip() for k, v in labels.items()}

    return model, labels

model, class_labels = load_assets()

# ---------------- UTILS ----------------
def normalize_label(label):
    return re.sub(r'\s+', ' ', label.strip().lower())

def composite_on_black(pil_img):
    if pil_img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", pil_img.size, (0, 0, 0))
        bg.paste(pil_img, mask=pil_img.split()[-1])
        return bg
    return pil_img.convert("RGB")

def preprocess_for_model(img):
    # 🔥 IMPORTANT: Must match training size (MobileNetV2 = 224x224)
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- STREAMLIT UI ----------------
st.title("🤟 Tamil ↔ Malayalam Sign Language Recognition")

uploaded_file = st.file_uploader(
    "Upload Tamil or Malayalam Sign Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        # Background removal
        no_bg = remove(image)
        cleaned_img = composite_on_black(no_bg)

        # Prediction
        x = preprocess_for_model(cleaned_img)
        preds = model.predict(x)
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        predicted_label = class_labels[class_idx]

        # -------- UI OUTPUT --------
        st.subheader("🔍 Prediction")
        st.write(f"**Predicted Class:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.4f}")

        # Text-to-Speech
        audio_path = "output.mp3"
        try:
            if re.search(r'[\u0D00-\u0D7F]', predicted_label):
                gTTS(predicted_label, lang="ml").save(audio_path)
            else:
                gTTS(predicted_label, lang="ta").save(audio_path)

            st.subheader("🔊 Pronunciation")
            st.audio(audio_path)
        except:
            st.warning("Voice generation failed.")

    except Exception as e:
        st.error("❌ An error occurred during processing")
        st.text(traceback.format_exc())
