import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("best_skin_model.keras")

# Disease names in the same order as LabelEncoder
class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# Image path from command line
img_path = sys.argv[1]

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\nPrediction")
print("----------------------")
print("Disease   :", class_names[predicted_class])
print(f"Confidence: {confidence:.2f}%")