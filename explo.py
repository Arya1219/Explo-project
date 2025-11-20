

import os
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


CSV_PATH = r"C:\Users\giria\OneDrive\Desktop\explo data.csv"
DATA_ROOT = r"C:\Users\giria\OneDrive\Desktop\DATA"



df_labels = pd.read_csv(CSV_PATH)
print("CSV columns:", df_labels.columns)
print(df_labels.head())


subject_col = 'subject_id'  # change if your CSV column name differs
label_col = 'label'         # change if needed


df_labels[label_col] = df_labels[label_col].astype(int) - 1  # Convert 1→0, 2→1
label_map = dict(zip(df_labels[subject_col].astype(str), df_labels[label_col].astype(int)))


records = []
all_images = glob(os.path.join(DATA_ROOT, "*.png")) + glob(os.path.join(DATA_ROOT, "*.jpg"))

for img_path in tqdm(all_images, desc="Mapping images"):
    filename = os.path.basename(img_path)
    subject_id = filename.split("_")[0]  # extract part before first underscore
    if subject_id in label_map:
        records.append({"subject_id": subject_id, "image_path": img_path, "label": label_map[subject_id]})

df = pd.DataFrame(records)
print(f" Found {len(df)} labeled images out of {len(all_images)} total")

# STEP 3: Train/Val/Test split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# STEP 4: Dataset class
IMG_SIZE = 224
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SpectrogramDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, int(row['label'])

train_loader = DataLoader(SpectrogramDataset(train_df, train_tf), batch_size=32, shuffle=True)
val_loader = DataLoader(SpectrogramDataset(val_df, test_tf), batch_size=32, shuffle=False)
test_loader = DataLoader(SpectrogramDataset(test_df, test_tf), batch_size=32, shuffle=False)

# STEP 5: CNN (ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
for p in model.parameters(): p.requires_grad = False
for p in model.layer4.parameters(): p.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# STEP 6: Train + Evaluate
def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_model(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    probs, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            probs += torch.softmax(out, dim=1)[:,1].cpu().numpy().tolist()
            labels += y.cpu().numpy().tolist()
    preds = (torch.tensor(probs) >= 0.5).int().numpy()
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": correct / len(loader.dataset),
        "f1": f1_score(labels, preds),
        "roc": roc_auc_score(labels, probs)
    }

# %%
# STEP 7: Training loop
best_f1 = 0
for epoch in range(1, 21):
    tr_loss, tr_acc = train_epoch(model, train_loader)
    val_metrics = eval_model(model, val_loader)
    print(f"Epoch {epoch}: Train Loss={tr_loss:.4f}, Acc={tr_acc:.3f} | Val Acc={val_metrics['acc']:.3f}, F1={val_metrics['f1']:.3f}, ROC={val_metrics['roc']:.3f}")
    if val_metrics["f1"] > best_f1:
        torch.save(model.state_dict(), "best_model.pth")
        best_f1 = val_metrics["f1"]
        print("Saved best model")

# STEP 8: Final Test
model.load_state_dict(torch.load("best_model.pth", map_location=device))
test_metrics = eval_model(model, test_loader)
print("=== TEST RESULTS ===")
print(test_metrics)

