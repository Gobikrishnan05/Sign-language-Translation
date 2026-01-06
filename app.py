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
PLACEHOLDER_IMAGE = r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png"
# ----------------------------------------

st.set_page_config(page_title="Tamil ↔ Malayalam Sign Recognition", layout="wide")

# ---------------- LOAD MODEL & LABELS ----------------
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    labels = {int(k): v.strip() for k, v in labels.items()}
    return model, labels

model, class_labels = load_assets()

# ---------------- TRANSLATION MAPS ----------------
MALAYALAM_TO_TAMIL_MAP = {
    "class_1  അ - A": "அ",
    "class_2  ആ - AA": "ஆ",
    "class_3  ഇ - I": "இ",
    "class_4  ഈ - II": "ஈ",
    "class_5  ഉ - U": "உ",
    "class_6  ഊ - UU": "ஊ",
    "class_7  ഋ - RU": "Can't find",
    "class_8  എ - E": "எ",
    "class_9  ഏ - EE": "ஏ",
    "class_10  ഐ - AI": "ஐ",
    "class_11  ഒ - O": "ஒ",
    "class_12  ഓ - OO": "ஓ",
    "class_13  ഔ - AU": "ஔ",
    "class_14  അം - AM": "Can't find",
    "class_15  അഃ - AHA": "Can't find"
}

TAMIL_TO_MALAYALAM_MAP = {
    "class_1  அ": "അ",
    "class_2  ஆ": "ആ",
    "class_3  இ": "ഇ",
    "class_4  ஈ": "ഈ",
    "class_5  உ": "ഉ",
    "class_6  ஊ": "ഊ",
    "class_7  எ": "എ",
    "class_8  ஏ": "ഏ",
    "class_9  ஐ": "ഐ",
    "class_10  ஒ": "ഒ",
    "class_11  ஓ": "ഓ",
    "class_12  ஔ": "ഔ",
    "class_13  ஃ": "അഃ"
}

# ---------------- REFERENCE IMAGES ----------------
reference_images = {
    "class_1 അ - A": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_01\frame_0001.jpg",
    "class_2 ആ - AA": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_02\frame_0001.jpg",
    "class_3 ഇ -  I": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png",
    "class_4 ഈ - II": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png",
    "class_5 ഉ - U": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_05\frame_0001.jpg",
    "class_6 ഊ - UU": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_06\frame_0001.jpg",
    "class_7 ഋ - RU": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png",
    "class_8 എ - E": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_03\frame_0001.jpg",
    "class_9 ഏ -  EE": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_04\frame_0001.jpg",
    "class_10 ഐ - AI": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_09\frame_0001.jpg",
    "class_11 ഓ - OO": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_10\frame_0001.jpg",
    "class_12 ഒ - O": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_11\frame_0001.jpg",
    "class_13 ഔ - AU": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Tamil_class_12\frame_0001.jpg",
    "class_14 അം - AM": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png",
    "class_15 അഃ - AHA": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png",
    "class_1  அ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_01\frame_0001.jpg",
    "class_2  ஆ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_02\frame_0001.jpg",
    "class_3  இ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_08\frame_0001.jpg",
    "class_4  ஈ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_09\frame_0001.jpg",
    "class_5  உ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_11\frame_0001.jpg",
    "class_6  ஊ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_12\frame_0001.jpg",
    "class_7  எ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png",
    "class_8  ஏ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png",
    "class_9  ஐ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_10\frame_0001.jpg",
    "class_10  ஒ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_11\frame_0001.jpg",
    "class_11  ஓ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\Malayalam_class_12\frame_0001.jpg",
    "class_12  ஔ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png",
    "class_13  ஃ": r"C:\Users\Gobi\OneDrive\Desktop\miniproject\BGremoved_clean_images\not sign.png"
}

# ---------------- UTILS ----------------
def normalize_label(label):
    return re.sub(r'\s+', ' ', label.strip().lower())

def get_translated_sign_name(raw_label):
    raw_label = raw_label.strip()
    if raw_label in MALAYALAM_TO_TAMIL_MAP:
        return MALAYALAM_TO_TAMIL_MAP[raw_label]
    elif raw_label in TAMIL_TO_MALAYALAM_MAP:
        return TAMIL_TO_MALAYALAM_MAP[raw_label]
    return re.sub(r"^class[_\s]*\d+\s*", "", raw_label).strip()

def composite_on_black(pil_img):
    if pil_img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", pil_img.size, (0, 0, 0))
        bg.paste(pil_img, mask=pil_img.split()[-1])
        return bg
    return pil_img.convert("RGB")

def preprocess_for_model(img):
    img = img.resize((128, 128))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- STREAMLIT UI ----------------
st.title("🤟 Bidirectional Tamil ↔ Malayalam Sign Language Recognition")

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
        translated_label = get_translated_sign_name(predicted_label)

        # -------- Translated Sign Image --------
        ref_image = None
        normalized_pred = normalize_label(predicted_label)

        for key, path in reference_images.items():
            if normalize_label(key) == normalized_pred and os.path.exists(path):
                ref_img = Image.open(path).convert("RGB")
                ref_img = composite_on_black(remove(ref_img))
                ref_image = ref_img
                break

        if ref_image is None:
            ref_image = Image.open(PLACEHOLDER_IMAGE).convert("RGB")

        # Voice output
        audio_path = "output.mp3"
        if re.search(r'[\u0D00-\u0D7F]', translated_label):
            gTTS(translated_label, lang="ml").save(audio_path)
        else:
            gTTS(translated_label, lang="ta").save(audio_path)

        # -------- UI OUTPUT --------
        st.subheader("🔍 Prediction")
        st.json({predicted_label: confidence})

        st.subheader("🔁 Translated Text")
        st.write(f"**{translated_label}**")

        st.subheader("🖐️ Translated Sign Image")
        st.image(ref_image, width=250)

        st.subheader("🔊 Pronunciation")
        st.audio(audio_path)

    except Exception:
        st.error("❌ An error occurred during processing")
        st.text(traceback.format_exc())
