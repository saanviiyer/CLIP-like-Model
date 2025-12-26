import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Dataset Class ---
class CTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"NSCLC": 0, "SCLC": 1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        label_str = self.annotations.iloc[index, 3]
        
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("L")
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return image, label

# --- 2. Architecture ---
class CTFoundationModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 256), nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        self.label_embedding = nn.Embedding(2, embed_dim)

    def forward(self, images, labels):
        img_emb = F.normalize(self.image_encoder(images), p=2, dim=1)
        lbl_emb = F.normalize(self.label_embedding(labels), p=2, dim=1)
        return img_emb, lbl_emb

# --- 3. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_loader = DataLoader(CTDataset('train/_annotations.csv', 'train/', transform), batch_size=32, shuffle=True)
test_loader = DataLoader(CTDataset('test/_annotations.csv', 'test/', transform), batch_size=32)

model = CTFoundationModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- 4. Training Loop ---
def train(epochs=10):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            img_emb, lbl_emb = model(imgs, lbls)
            
            # Contrastive Loss: Maximize cosine similarity
            loss = 1 - torch.mean(torch.sum(img_emb * lbl_emb, dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

# --- 5. Validation & Confusion Matrix ---
def validate(loader, model):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        # Pre-calculate normalized class embeddings (the "prompts")
        class_embeddings = F.normalize(model.label_embedding(torch.tensor([0, 1]).to(device)), p=2, dim=1)

        for images, labels in loader:
            images = images.to(device)
            img_emb = F.normalize(model.image_encoder(images), p=2, dim=1)

            # Similarity search: Image @ Labels^T
            logits = img_emb @ class_embeddings.t()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nTEST DATASET PERFORMANCE\n" + "="*25)
    print(classification_report(all_labels, all_preds, target_names=["NSCLC", "SCLC"]))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["NSCLC", "SCLC"], yticklabels=["NSCLC", "SCLC"])
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

# Run
train(10)
validate(test_loader, model)