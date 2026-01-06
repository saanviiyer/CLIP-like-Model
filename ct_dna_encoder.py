# Independent CT and DNA sequence encoders
# To encoder DNA Sequences, using the DNABert architecture

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
import numpy as np

class CTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {
            "NSCLC": 0,
            "SCLC": 1
        }

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

class CTEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, images):
        img_emb = F.normalize(self.image_encoder(images), p=2, dim=1)
        return img_emb

class DNAEncoder(nn.Module):
    def __init__(self, embed_dim=128, seq_length=512):
        super().__init__()
        self.seq_length = seq_length
        
        self.dna_model = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.projection = nn.Linear(256, embed_dim)
        
    def one_hot_encode(self, sequences):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        batch_size = len(sequences)
        
        encoded = torch.zeros(batch_size, 4, self.seq_length)
        for i, seq in enumerate(sequences):
            for j, nucleotide in enumerate(seq[:self.seq_length].upper()):
                if nucleotide in mapping:
                    encoded[i, mapping[nucleotide], j] = 1
        return encoded
    
    def forward(self, sequences):
        if isinstance(sequences, list):
            sequences = self.one_hot_encode(sequences)
        
        sequences = sequences.to(next(self.parameters()).device)
        
        dna_features = self.dna_model(sequences)
        dna_emb = F.normalize(self.projection(dna_features), p=2, dim=1)
        
        return dna_emb

class MultimodalCTDNAModel(nn.Module):
    def __init__(self, embed_dim=128, seq_length=512):
        super().__init__()
        self.ct_encoder = CTEncoder(embed_dim=embed_dim)
        self.dna_encoder = DNAEncoder(embed_dim=embed_dim, seq_length=seq_length)
        self.label_embedding = nn.Embedding(2, embed_dim)
        
    def forward(self, images=None, dna_sequences=None, labels=None):
        embeddings = []
        
        if images is not None:
            ct_emb = self.ct_encoder(images)
            embeddings.append(ct_emb)
        
        if dna_sequences is not None:
            dna_emb = self.dna_encoder(dna_sequences)
            embeddings.append(dna_emb)
        
        if labels is not None:
            lbl_emb = F.normalize(self.label_embedding(labels), p=2, dim=1)
            return embeddings, lbl_emb
        
        return embeddings

def train_ct_only(model, train_loader, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_weights = torch.tensor([1.0, 3.7], device=device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            class_embeddings = F.normalize(
                model.label_embedding(torch.tensor([0, 1]).to(device)),
                p=2,
                dim=1
            )
            
            ct_emb = model.ct_encoder(imgs)
            
            logits = ct_emb @ class_embeddings.t() * 10.0
            
            loss = F.cross_entropy(logits, lbls, weight=class_weights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%")

def generate_synthetic_dna(batch_size, seq_length=512):
    nucleotides = ['A', 'C', 'G', 'T']
    sequences = []
    for _ in range(batch_size):
        seq = ''.join(np.random.choice(nucleotides, seq_length))
        sequences.append(seq)
    return sequences

def train_dna(model, device, epochs=5, batch_size=64, num_batches=20):
    optimizer = torch.optim.Adam(model.dna_encoder.parameters(), lr=1e-3)
    
    print("\n" + "="*50)
    print("TRAINING DNA ENCODER (with synthetic data)")
    print("="*50)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx in range(num_batches):
            dna_seqs = generate_synthetic_dna(batch_size)
            lbls = torch.randint(0, 2, (batch_size,)).to(device)
            
            class_embeddings = F.normalize(
                model.label_embedding(torch.tensor([0, 1]).to(device)),
                p=2,
                dim=1
            )
            
            dna_emb = model.dna_encoder(dna_seqs)
            
            logits = dna_emb @ class_embeddings.t() * 10.0
            
            loss = F.cross_entropy(logits, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
        
        avg_loss = total_loss / num_batches
        accuracy = 100 * correct / total
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%")

def validate(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        class_embeddings = F.normalize(
            model.label_embedding(torch.tensor([0, 1]).to(device)),
            p=2,
            dim=1
        )
        
        for images, labels in loader:
            images = images.to(device)
            ct_emb = model.ct_encoder(images)
            
            logits = ct_emb @ class_embeddings.t() * 10.0
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print("\n" + "="*50)
    print("TEST DATASET PERFORMANCE (CT ENCODER)")
    print("="*50)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=[
            "NSCLC (Non-Small Cell Lung Cancer)",
            "SCLC (Small Cell Lung Cancer)"
        ],
        zero_division=0
    ))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["NSCLC", "SCLC"],
        yticklabels=["NSCLC", "SCLC"]
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - CT Encoder')
    plt.tight_layout()
    plt.show()
    
    unique, counts = torch.tensor(all_preds).unique(return_counts=True)
    print("\nPrediction distribution:")
    for cls, cnt in zip(unique.numpy(), counts.numpy()):
        print(f"  Class {cls}: {cnt} predictions ({100*cnt/len(all_preds):.1f}%)")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_loader = DataLoader(
        CTDataset('train/_annotations.csv', 'train/', transform),
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        CTDataset('test/_annotations.csv', 'test/', transform),
        batch_size=64,
        num_workers=2,
        pin_memory=True
    )
    
    model = MultimodalCTDNAModel(embed_dim=128, seq_length=512).to(device)
    
    print("="*50)
    print("TRAINING CT ENCODER")
    print("="*50)
    train_ct_only(model, train_loader, device, epochs=5)
    
    validate(test_loader, model, device)
    
    train_dna(model, device, epochs=3)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
