import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---- PATHS ----
model_path = r"C:\Users\giria\best_model.pth"       
image_dir = r"C:\Users\giria\OneDrive\Desktop\DATA"  
sample_csv = r"C:\Users\giria\OneDrive\Desktop\explo data.csv"
output_csv = r"C:\Users\giria\OneDrive\Desktop\predictions_with_accuracy.csv"

# ---- LOAD MODEL ----
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ---- IMAGE PREPROCESSING ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- LOAD TRUE LABELS ----
true_df = pd.read_csv(sample_csv)
true_df['subject_id'] = true_df['subject_id'].astype(str)
true_labels = {}
for _, row in true_df.iterrows():
    true_labels[str(row['subject_id'])] = int(row['label'])

# ---- PREDICT ----
results = []
y_true, y_pred = [], []

print("🔍 Starting predictions... Please wait.")
for img_name in tqdm(os.listdir(image_dir), desc="Predicting"):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_id = img_name.split('.')[0]  # remove .png
    img_path = os.path.join(image_dir, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {img_name}: {e}")
        continue

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, 1).item()
        confidence = probs[0][predicted].item()

    label = "Autism" if predicted == 1 else "Non-Autism"
    results.append({
        "image_name": img_name,
        "prediction": label,
        "confidence": round(confidence, 4)
    })

    # Record true and predicted for accuracy
    if img_id in true_labels:
        y_true.append(true_labels[img_id])
        y_pred.append(predicted)

# ---- SAVE RESULTS ----
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

# ---- CALCULATE ACCURACY ----
if len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n✅ Accuracy: {acc*100:.2f}% | F1 Score: {f1:.3f}")
    print("Confusion Matrix:\n", cm)
else:
    print("⚠️ No matching labels found for accuracy calculation.")

print(f"\n📁 Predictions saved to:\n{output_csv}")

