# app.py
from huggingface_hub import hf_hub_download
import streamlit as st
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import tensorflow 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="Muthuswamy/catvsdog1",  # 👈 your path
        filename="cat Vs dod model.h5"
    )
    return tensorflow.keras.models.load_model(path)

model = load_model()

# ── Predict Function ───────────────────────────────────
def predict(url, model, target_size=(128,128)):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer' : 'https://www.google.com/'
    }
    try:
        res = requests.get(url, timeout=10, headers=headers)
        res.raise_for_status()

        img = Image.open(BytesIO(res.content))
        img = img.convert('RGB')
        img = img.resize(target_size)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])

        label = 'Dog' if confidence >= 0.5 else 'Cat'
        confidence_pct = confidence * 100 if confidence >= 0.5 else (1 - confidence) * 100

        return label, confidence_pct, img

    except Exception as e:
        return None, None, str(e)

# ── Streamlit UI ───────────────────────────────────────
st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Enter an image URL to classify")

# URL input
url = st.text_input("Image URL", placeholder="https://example.com/dog.jpg")

if st.button("Predict"):
    if url.strip() == "":
        st.warning("⚠️ Please enter a URL")
    
    else:
        with st.spinner("Classifying..."):
            label, confidence, img = predict(url, model)
        
        if label is None:
            st.error(f"❌ Error: {img}")   # img contains error msg here
        
        else:
            # Show image
            st.image(img, caption="Input Image", use_column_width=True)
            
            # Show result
            if label == 'Dog':
                st.success(f"🐶 Prediction: **{label}**")
            else:
                st.success(f"🐱 Prediction: **{label}**")
            
            # Show confidence bar
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(confidence / 100)
