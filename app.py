import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = r"C:\Users\giria\best_model.pth"

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🧠 Autism Detection from Time-Frequency Spectrogram")
st.markdown("Upload a **STFT spectrogram image** to classify Autism vs Non-Autism.")

uploaded_file = st.file_uploader("Choose an image (.png, .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, 1).item()
        confidence = probs[0][predicted].item()

    label = "🩺 Autism" if predicted == 1 else "🙂 Non-Autism"
    st.markdown(f"### **Prediction:** {label}")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
