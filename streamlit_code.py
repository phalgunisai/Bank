import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew, entropy
import preprocessing as pp
import tempfile
import os
import pygame  # Import pygame for playing audio

# Initialize the pygame mixer
pygame.mixer.init()

# Load and preprocess the dataset for authenticity detection
data = pd.read_csv('banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
target_count = data.auth.value_counts()
nb_to_delete = target_count[0] - target_count[1]
data = data[nb_to_delete:]

x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
clf.fit(x_train, y_train.values.ravel())

# Load the PyTorch model for denomination classification
MODEL_PATH = "models/final.pt"

def load_model():
    import torch.nn as nn
    import torchvision.models as models

    # Define the model structure
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 7)  # Assuming 7 classes, modify if needed
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    except FileNotFoundError:
        st.error(f"Model file {MODEL_PATH} not found.")
        return None
    return model

model = load_model()

# Define image transformations for PyTorch model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit interface
st.title("Banknote Authentication and Denomination Classification")

# Options for input source
input_source = st.radio("Select Image Source", ("Gallery", "Camera"))

image = None
if input_source == "Gallery":
    uploaded_file = st.file_uploader("Upload an image of a banknote", type=None)  # Accept any image type
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

elif input_source == "Camera":
    if hasattr(st, "camera_input"):
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(camera_image.getvalue())
            image = Image.open(temp_file.name).convert('RGB')
            os.unlink(temp_file.name)
    else:
        st.warning("Camera input is not available in your Streamlit version.")

if image:
    try:
        # Display the image
        st.image(image, caption="Selected Image", use_column_width=True)

        # Convert image for authenticity check
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        norm_image = np.array(opencv_image, dtype=np.float32) / 255.0

        # Compute features for authenticity check
        var = np.var(norm_image, axis=None)
        sk = skew(norm_image, axis=None)
        kur = kurtosis(norm_image, axis=None)
        ent = entropy(norm_image, axis=None) / 100

        # Validate features
        if not np.isfinite(var) or not np.isfinite(sk) or not np.isfinite(kur) or not np.isfinite(ent):
            st.error("Error: Computed features contain invalid values. Please use a valid image.")
        else:
            # Predict authenticity
            result = clf.predict(np.array([[var, sk, kur, ent]]))
            authenticity = "Real Currency" if result[0] == 0 else "Fake Currency"
            st.success(f"Authenticity: {authenticity}")

            # Step 2: Denomination Classification
            input_image = transform(image).unsqueeze(0)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_image = input_image.to(device)
            model.to(device)

            with torch.no_grad():
                outputs = model(input_image)
                _, predicted = torch.max(outputs, 1)
                predicted_class = pp.class_names[predicted.item()]

            st.success(f"Denomination: {predicted_class}")

            # Path to the audio folder (update to match the correct folder structure in the repo)
            audio_path = "audio"  # Adjust the path to point to the 'audio' folder in your repo

            # Play corresponding sound in the background using pygame
            if authenticity == "Real Currency":
                audio_file = os.path.join(audio_path, f"{predicted_class}_real.mp3")
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
            else:
                audio_file = os.path.join(audio_path, f"{predicted_class}_fake.mp3")
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
