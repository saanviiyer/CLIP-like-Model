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
import random

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
        return F.normalize(self.image_encoder(images), p=2, dim=1)


class DNAEncoder(nn.Module):
    def __init__(self, embed_dim=128, seq_length=512):
        super().__init__()
        self.seq_length = seq_length
        
        self.dna_model = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.projection = nn.Linear(256, embed_dim)
        
    def forward(self, x):
        # input as (batch, seq_len) with integer values for A, C, G, T nucleotides as 0, 1, 2, 3
        x = F.one_hot(x, num_classes=4).float()  # (batch, seq_len, 4)
        x = x.permute(0, 2, 1)
        dna_features = self.dna_model(x)
        return F.normalize(self.projection(dna_features), p=2, dim=1)


class MultimodalCTDNAModel(nn.Module):
    # combining encoders
    def __init__(self, embed_dim=128, seq_length=512):
        super().__init__()
        self.ct_encoder = CTEncoder(embed_dim=embed_dim)
        self.dna_encoder = DNAEncoder(embed_dim=embed_dim, seq_length=seq_length)
        self.label_embedding = nn.Embedding(2, embed_dim)


# generating fake tumors
def add_black_tumor_to_ct(ct_image, tumor_type, position=None):
    """
    Add a BLACK geometric shape (fake tumor) to a real CT image.
    
    Args:
        ct_image: Normalized CT tensor (1, H, W)
        tumor_type: 0=Circle, 1=Square, 2=Triangle, 3=Star
        position: (y, x) or None for random placement
    """
    img = ct_image.clone()
    H, W = img.shape[1], img.shape[2]
    
    if position is None:
        y = random.randint(20, H - 40)
        x = random.randint(20, W - 40)
    else:
        y, x = position
    
    # BLACK shapes (value = -1.0 in normalized images)
    if tumor_type == 0:  # Circle
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                if dy**2 + dx**2 <= 100 and 0 <= y+dy < H and 0 <= x+dx < W:
                    img[0, y+dy, x+dx] = -1.0
    
    elif tumor_type == 1:  # Square
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                if 0 <= y+dy < H and 0 <= x+dx < W:
                    img[0, y+dy, x+dx] = -1.0
    
    elif tumor_type == 2:  # Triangle
        for dy in range(0, 15):
            for dx in range(-dy, dy+1):
                if 0 <= y+dy < H and 0 <= x+dx < W:
                    img[0, y+dy, x+dx] = -1.0
    
    elif tumor_type == 3:  # Star/Cross
        for dy in range(-12, 13):
            for dx in range(-12, 13):
                if (abs(dy) < 3 or abs(dx) < 3) and 0 <= y+dy < H and 0 <= x+dx < W:
                    img[0, y+dy, x+dx] = -1.0
    
    return img


def generate_tumor_dna_signature(tumor_type, seq_length=512):
    # create unique DNA pattern for each tumor type.

    np.random.seed(tumor_type * 1000)
    nucleotides = ['A', 'C', 'G', 'T']
    
    if tumor_type == 0:  # Circle → A/T-rich
        seq = ''.join(np.random.choice(['A', 'T'], seq_length, p=[0.6, 0.4]))
    elif tumor_type == 1:  # Square → C/G-rich
        seq = ''.join(np.random.choice(['C', 'G'], seq_length, p=[0.6, 0.4]))
    elif tumor_type == 2:  # Triangle → Repeating ACGT
        seq = ('ACGT' * (seq_length // 4))[:seq_length]
    elif tumor_type == 3:  # Star → High GC content
        seq = ''.join(np.random.choice(nucleotides, seq_length, p=[0.1, 0.4, 0.4, 0.1]))
    else:
        seq = ''.join(np.random.choice(nucleotides, seq_length))
    
    np.random.seed(None)
    return seq


class RealCTWithFakeTumorsDataset(Dataset):
   # add tumor shapes to ct scans
    def __init__(self, ct_dataset, num_tumor_types=4, tumor_probability=0.7):
        self.ct_dataset = ct_dataset
        self.num_tumor_types = num_tumor_types
        self.tumor_probability = tumor_probability
        
        # Pre-generate DNA signatures for each tumor type
        self.tumor_signatures = {
            i: generate_tumor_dna_signature(i, seq_length=512) 
            for i in range(num_tumor_types)
        }
        
        # Random DNA for non-tumor images
        self.no_tumor_dna = ''.join(np.random.choice(['A', 'C', 'G', 'T'], 512))
    
    def __len__(self):
        return len(self.ct_dataset)
    
    def __getitem__(self, idx):
        ct_image, original_label = self.ct_dataset[idx]
        
        # Randomly decide to add fake tumor(s)
        if random.random() < self.tumor_probability:
            num_tumors = random.randint(1, 3)
            tumor_types = random.sample(range(self.num_tumor_types), min(num_tumors, self.num_tumor_types))
            
            # Add BLACK shapes to the real CT image
            modified_ct = ct_image.clone()
            for t_type in tumor_types:
                modified_ct = add_black_tumor_to_ct(modified_ct, t_type)
            
            # Use DNA signature of PRIMARY tumor
            primary_tumor = tumor_types[0]
            dna_seq = self.tumor_signatures[primary_tumor]
            tumor_label = primary_tumor
        else:
            # No fake tumor - original image
            modified_ct = ct_image
            dna_seq = self.no_tumor_dna
            tumor_label = -1
        
        return modified_ct, dna_seq, tumor_label, original_label

def dna_string_to_tensor(dna_sequences, seq_length=512):
    # converting DNA strings (ACGT) to integer tensors (0123) to speed up computing
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    batch_size = len(dna_sequences)
    tensor = torch.zeros(batch_size, seq_length, dtype=torch.long)
    
    for i, seq in enumerate(dna_sequences):
        for j, nucleotide in enumerate(seq[:seq_length].upper()):
            if nucleotide in mapping:
                tensor[i, j] = mapping[nucleotide]
    
    return tensor


def clip_contrastive_loss(ct_embeddings, dna_embeddings, temperature=0.07):
    batch_size = ct_embeddings.shape[0]
    logits = (ct_embeddings @ dna_embeddings.t()) / temperature
    labels = torch.arange(batch_size, device=ct_embeddings.device)
    
    loss_ct = F.cross_entropy(logits, labels)
    loss_dna = F.cross_entropy(logits.t(), labels)
    
    return (loss_ct + loss_dna) / 2


def train_clip_on_real_ct(model, train_loader, device, epochs=30, temperature=0.07):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("\n" + "="*60)
    print("TRAINING CLIP: REAL CT SCANS + FAKE BLACK TUMORS")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for images, dna_seqs, tumor_labels, original_labels in train_loader:
            images = images.to(device)
            dna_tensor = dna_string_to_tensor(dna_seqs, seq_length=512).to(device)
            
            # Get embeddings
            ct_emb = model.ct_encoder(images)
            dna_emb = model.dna_encoder(dna_tensor)
            
            # CLIP loss
            loss = clip_contrastive_loss(ct_emb, dna_emb, temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} | CLIP Loss: {avg_loss:.4f}")
    
    print("\nCLIP training complete!\n")


# ============================================================================
# PART 5: EVALUATION
# ============================================================================

def evaluate_fake_tumor_isolation(model, test_loader, device, num_tumor_types=4):
    model.eval()
    
    print("="*60)
    print("EVALUATING FAKE TUMOR ISOLATION")
    print("="*60)
    
    # Pre-compute DNA signature embeddings
    tumor_signatures = {i: generate_tumor_dna_signature(i, 512) for i in range(num_tumor_types)}
    
    with torch.no_grad():
        signature_embeddings = []
        for t_type in range(num_tumor_types):
            sig_tensor = dna_string_to_tensor([tumor_signatures[t_type]], seq_length=512).to(device)
            dna_emb = model.dna_encoder(sig_tensor)
            signature_embeddings.append(dna_emb)
        signature_embeddings = torch.cat(signature_embeddings, dim=0)
        
        correct, total = 0, 0
        
        for images, dna_seqs, tumor_labels, original_labels in test_loader:
            images = images.to(device)
            tumor_labels = torch.tensor(tumor_labels).to(device)
            
            ct_emb = model.ct_encoder(images)
            similarities = ct_emb @ signature_embeddings.t()
            predictions = torch.argmax(similarities, dim=1)
            
            has_tumor = tumor_labels != -1
            if has_tumor.sum() > 0:
                correct += (predictions[has_tumor] == tumor_labels[has_tumor]).sum().item()
                total += has_tumor.sum().item()
        
        if total > 0:
            accuracy = 100 * correct / total
            print(f"\nFake Tumor Classification Accuracy: {accuracy:.2f}%")
            print(f"Correct: {correct}/{total}")
            print(f"\n{'='*60}")
            print("INTERPRETATION:")
            print(f"{'='*60}")
            print("✓ Random guess: ~25% (4 tumor types)")
            print(f"✓ Achieved: {accuracy:.2f}%")
            print("✓ High accuracy means CLIP learned tumor-DNA associations!")
        else:
            print("No fake tumors found in test set!")
    
    visualize_predictions(model, test_loader, device, signature_embeddings, num_examples=8)


def visualize_predictions(model, test_loader, device, signature_embeddings, num_examples=8):
    """Visualize REAL CT scans with BLACK fake tumors and model predictions."""
    model.eval()
    
    images_list, labels_list, preds_list = [], [], []
    
    with torch.no_grad():
        for images, dna_seqs, tumor_labels, original_labels in test_loader:
            if len(images_list) >= num_examples:
                break
            
            images_batch = images.to(device)
            ct_emb = model.ct_encoder(images_batch)
            similarities = ct_emb @ signature_embeddings.t()
            predictions = torch.argmax(similarities, dim=1)
            
            for i in range(min(num_examples - len(images_list), len(images))):
                if tumor_labels[i] != -1:
                    images_list.append(images[i])
                    labels_list.append(tumor_labels[i])
                    preds_list.append(predictions[i].cpu().item())
    
    if len(images_list) == 0:
        print("No tumor images to visualize!")
        return
    
    # Plot
    num_cols = min(4, len(images_list))
    num_rows = (len(images_list) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    tumor_names = ['Circle', 'Square', 'Triangle', 'Star']
    
    for idx, (img, true_label, pred_label) in enumerate(zip(images_list, labels_list, preds_list)):
        axes[idx].imshow(img[0], cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(
            f'True: {tumor_names[true_label]}\nPred: {tumor_names[pred_label]}',
            color=color, fontweight='bold'
        )
        axes[idx].axis('off')
    
    for idx in range(len(images_list), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('real_ct_fake_tumor_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Visualization saved: 'real_ct_fake_tumor_predictions.png'")
    print("✓ Look for BLACK shapes on REAL CT scans!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Load REAL CT scan data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    print("Loading REAL CT scan datasets...")
    real_train = CTDataset('train/_annotations.csv', 'train/', transform)
    real_test = CTDataset('test/_annotations.csv', 'test/', transform)
    
    # Add fake tumors to real CT scans
    print("Augmenting CT scans with FAKE BLACK tumors...")
    train_with_tumors = RealCTWithFakeTumorsDataset(real_train, num_tumor_types=4, tumor_probability=0.7)
    test_with_tumors = RealCTWithFakeTumorsDataset(real_test, num_tumor_types=4, tumor_probability=0.7)
    
    train_loader = DataLoader(train_with_tumors, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_with_tumors, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    print("Initializing multimodal CT-DNA model...")
    model = MultimodalCTDNAModel(embed_dim=128, seq_length=512).to(device)
    
    # Train CLIP
    train_clip_on_real_ct(model, train_loader, device, epochs=30)
    
    # Evaluate
    evaluate_fake_tumor_isolation(model, test_loader, device, num_tumor_types=4)
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)