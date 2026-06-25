# Autism Detection from EEG STFT Spectrograms using Deep Learning

## Overview

This project presents a deep learning-based framework for detecting Autism Spectrum Disorder (ASD) using EEG-derived Short-Time Fourier Transform (STFT) spectrogram images. Instead of manually engineering EEG features, the system employs transfer learning with ResNet18 to automatically learn discriminative representations from spectrogram images.

The project includes:

- Model training using PyTorch
- Batch prediction on unseen spectrogram images
- Interactive Streamlit web application for inference

---

## Features

- Transfer Learning using ResNet18
- Automatic subject-label mapping
- Custom PyTorch Dataset
- Data augmentation
- Binary Autism classification
- Confidence score estimation
- Batch prediction utility
- CSV prediction export
- Streamlit deployment

---

## Dataset

The dataset consists of:

- EEG STFT Spectrogram Images (.png/.jpg)
- CSV file containing:

```
subject_id
label
```

Labels are converted from

```
1 → Non Autism
2 → Autism
```

to

```
0 → Non Autism
1 → Autism
```

---

## Methodology

1. Read subject labels from CSV.
2. Match each spectrogram image with its subject ID.
3. Create a labeled dataset.
4. Split data into Train, Validation, and Test sets.
5. Apply image preprocessing and augmentation.
6. Fine-tune a pretrained ResNet18 model.
7. Evaluate performance using Accuracy and F1 Score.
8. Save the trained model.
9. Deploy using Streamlit.

---

## Model Architecture

- Backbone: ResNet18
- Transfer Learning
- Frozen early layers
- Fine-tuned Layer4
- Fully connected output layer modified for binary classification

---

## Image Preprocessing

Training:

- Resize (224×224)
- Random Horizontal Flip
- ToTensor
- ImageNet Normalization

Testing:

- Resize
- ToTensor
- ImageNet Normalization

---

## Project Structure

```
.
├── app.py
├── explo.py
├── predict_autism.py
├── final_predictions.csv
├── best_model.pth
├── DATA/
├── README.md
```

---

## Installation

```bash
git clone <repository-url>

cd Autism-Detection

pip install -r requirements.txt
```

---

## Training

```bash
python explo.py
```

---

## Batch Prediction

```bash
python predict_autism.py
```

Predictions are stored in

```
predictions_with_accuracy.csv
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

Upload a spectrogram image and receive:

- Predicted Class
- Confidence Score

---

## Technologies

- Python
- PyTorch
- TorchVision
- Streamlit
- Pandas
- NumPy
- Pillow
- scikit-learn
- tqdm

---

## Future Work

- Subject-wise cross-validation
- EfficientNet and Vision Transformer comparison
- Grad-CAM explainability
- Cloud deployment
- Multi-class neurological disorder classification

---

## Author

Arya Giri

Indian Institute of Technology (BHU), Varanasi
