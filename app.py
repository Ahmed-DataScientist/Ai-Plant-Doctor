import streamlit as st
from ultralytics import YOLO
import json
from PIL import Image

# Load disease info JSON file
with open("C:/Users/Hp/OneDrive/Desktop/fyp navtac/disease_info.json", "r") as f:
    disease_info = json.load(f)

# Load YOLO model
model = YOLO("model.pt")  # Replace with the actual path to your trained YOLO model

# Streamlit app
st.title("AI Plant Doctor")
st.header("Diseases and Treatments")

# Upload image
uploaded_image = st.file_uploader("Upload an image of a leaf", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    # Run inference on the uploaded image
    results = model(image)

    # Get the predicted class name
    class_id = int(results[0].boxes.cls[0].item())  # Assuming one object per image
    class_name = model.names[class_id]  # Get the class name based on the model's prediction

    st.success(f"🧪 **Prediction:** {class_name}")

    # Normalize the class name to match the keys in the disease_info dictionary
    normalized_class = class_name.strip().title()  # Normalize class name to match JSON keys

    # Show symptoms and treatment if the class name matches
    if normalized_class in disease_info:
        info = disease_info[normalized_class]
        st.markdown("### 🩺 Symptoms")
        st.write(info["symptoms"])
        st.markdown("### 💊 Treatment")
        st.write(info["treatment"])
    else:
        st.warning("No additional information found for this disease.")
