import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# --- Auto-detect folder where this script is located ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model file name ---
MODEL_FILE = "gesture_model_mobilenetv2.h5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

# --- Test image file name ---
TEST_IMAGE_FILE =  r"C:\Users\samra\OneDrive\Desktop\My codes\Prodigy_task4\frame_08_03_0004.png"
IMAGE_PATH = os.path.join(BASE_DIR, TEST_IMAGE_FILE)

# --- Check model file ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# --- Check image file ---
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

# --- Load the model ---
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- Define your gesture labels (EDIT this to match your dataset) ---
labels = ['fist', 'palm', 'peace', 'thumbs_up', 'okay']  # Example labels

# --- Load and preprocess image ---
img = image.load_img(IMAGE_PATH, target_size=(160, 160))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# --- Predict ---
pred = model.predict(img_array)
predicted_class = labels[np.argmax(pred)]
confidence = np.max(pred) * 100

# --- Show result in console ---
print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")

# --- Display image with prediction ---
plt.imshow(image.load_img(IMAGE_PATH))
plt.axis('off')
plt.title(f"{predicted_class} ({confidence:.2f}%)", fontsize=14, color='green')
plt.show()