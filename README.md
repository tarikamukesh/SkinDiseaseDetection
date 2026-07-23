# Skin Lesion Detection using Deep Learning

## Project Overview

This project is a deep learning-based skin lesion classification system developed using the HAM10000 dataset. It classifies dermatoscopic images into seven different types of skin lesions using the MobileNetV2 transfer learning model.

---

## Features

- Load and preprocess the HAM10000 dataset
- Encode disease labels
- Split dataset into training, validation, and testing sets
- Image preprocessing using TensorFlow
- Transfer Learning using MobileNetV2
- Predict skin lesion type from an input image
- Save and reuse the trained model

---

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Pandas
- NumPy
- Scikit-learn
- Pillow
- Matplotlib

---

## Dataset

**Dataset:** HAM10000 (Human Against Machine with 10000 Training Images)

Number of Images: **10,015**

Disease Classes:

- akiec – Actinic Keratoses
- bcc – Basal Cell Carcinoma
- bkl – Benign Keratosis-like Lesions
- df – Dermatofibroma
- mel – Melanoma
- nv – Melanocytic Nevi
- vasc – Vascular Lesions

---

## Model Architecture

- MobileNetV2 (Pretrained on ImageNet)
- GlobalAveragePooling2D
- Dropout (0.3)
- Dense (128, ReLU)
- Dense (7, Softmax)

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **76.45%** |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 10 |

---

## Project Structure

```
SkinDiseaseDetection/
│
├── dataset/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
│
├── train.py
├── predict.py
├── best_skin_model.keras
├── skin_disease_model.keras
├── requirements.txt
└── README.md
```

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Training the Model

```bash
python train.py
```

---

## Predicting a Skin Lesion

```bash
python predict.py path_to_image.jpg
```

Example:

```bash
python predict.py dataset/HAM10000_images_part_1/ISIC_0024306.jpg
```

---

## Future Improvements

- Fine-tune MobileNetV2 by unfreezing some layers
- Add data augmentation
- Improve class balancing
- Deploy as a web application using Streamlit or Flask
- Improve accuracy with hyperparameter tuning

---

## Author

**Tarika Mukesh**

Deep Learning Project using TensorFlow and MobileNetV2.