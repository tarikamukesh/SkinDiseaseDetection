import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load metadata
metadata = pd.read_csv("dataset/HAM10000_metadata.csv")

print("Dataset loaded successfully!")
print("Shape:", metadata.shape)

# Encode labels
label_encoder = LabelEncoder()
metadata["label"] = label_encoder.fit_transform(metadata["dx"])

print("\nDisease Label Mapping:")

for disease, label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{disease} --> {label}")

print("\nFirst 5 rows:")
print(metadata[["image_id", "dx", "label"]].head())
from sklearn.model_selection import train_test_split

# Split into training (80%) and temporary (20%)
train_df, temp_df = train_test_split(
    metadata,
    test_size=0.2,
    random_state=42,
    stratify=metadata["label"]
)

# Split temporary into validation (10%) and test (10%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

print("\nDataset Split")
print("Training images :", len(train_df))
print("Validation images:", len(val_df))
print("Testing images   :", len(test_df))
import glob

# Create a dictionary of image_id -> full image path
image_paths = {}

for img in glob.glob("dataset/HAM10000_images_part_1/*.jpg"):
    image_id = img.split("\\")[-1].replace(".jpg", "")
    image_paths[image_id] = img

for img in glob.glob("dataset/HAM10000_images_part_2/*.jpg"):
    image_id = img.split("\\")[-1].replace(".jpg", "")
    image_paths[image_id] = img

# Add image paths to each dataframe
train_df["image_path"] = train_df["image_id"].map(image_paths)
val_df["image_path"] = val_df["image_id"].map(image_paths)
test_df["image_path"] = test_df["image_id"].map(image_paths)

print("\nSample Training Data:")
print(train_df[["image_id", "image_path", "label"]].head())

print("\nMissing train images:", train_df["image_path"].isnull().sum())
print("Missing validation images:", val_df["image_path"].isnull().sum())
print("Missing test images:", test_df["image_path"].isnull().sum())
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label
# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["image_path"].values, train_df["label"].values)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["image_path"].values, val_df["label"].values)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_df["image_path"].values, test_df["label"].values)
)

# Apply preprocessing
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and optimize
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("\n✅ TensorFlow datasets created successfully!")
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the pretrained layers
base_model.trainable = False

# Build the classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
output = Dense(7, activation="softmax")(x)

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Show model summary
model.summary()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n✅ Model compiled successfully!")
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Stop training if validation loss doesn't improve
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Save the best model
checkpoint = ModelCheckpoint(
    "best_skin_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)

# Save the final model
model.save("skin_disease_model.keras")

print("\n🎉 Training completed successfully!")
print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
import matplotlib.pyplot as plt

# Accuracy graph
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.show()

# Loss graph
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()