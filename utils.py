import os
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

class LungDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, model_name, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Menentukan ukuran gambar berdasarkan config model (biasanya 224)
        self.size = (
            self.processor.size["shortest_edge"],
            self.processor.size["shortest_edge"]
        ) if "shortest_edge" in self.processor.size else (224, 224)
        
        # Normalisasi standar ImageNet
        self.normalize = transforms.Normalize(
            mean=self.processor.image_mean, 
            std=self.processor.image_std
        )

    def setup(self, stage=None):
        # Augmentasi data untuk training
        train_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            self.normalize
        ])

        # Transformasi standar untuk validasi/test (tanpa augmentasi acak)
        val_test_transform = transforms.Compose([
            transforms.Resize(self.size), 
            transforms.ToTensor(),
            self.normalize
        ])

        self.train_ds = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=train_transform)
        self.val_ds = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=val_test_transform)
        self.test_ds = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=val_test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

def get_class_weights(data_dir):
    """
    Menghitung class weights dan menampilkan statistik distribusi data.
    Berguna untuk melihat apakah dataset imbalanced.
    """
    train_dir = os.path.join(data_dir, 'train')
    print(f"\n[INFO] Checking class distribution in: {train_dir}")
    
    # Scanning dataset untuk mendapatkan label
    # Warning: Pada dataset jutaan gambar, ini bisa lambat.
    dataset = datasets.ImageFolder(train_dir)
    targets = dataset.targets
    classes = dataset.classes
    
    # Hitung jumlah sampel per kelas
    unique, counts = np.unique(targets, return_counts=True)
    total_samples = sum(counts)
    
    # Hitung bobot penyeimbang (balanced weights)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    
    # --- TAMPILKAN TABEL STATISTIK ---
    print(f"[INFO] Total samples: {total_samples}")
    print("-" * 55)
    print(f"{'Class Name':<20} | {'Count':<10} | {'Weight':<10}")
    print("-" * 55)
    
    for i, class_name in enumerate(classes):
        # Pastikan index aman jika ada kelas kosong (jarang terjadi)
        count = counts[i] if i < len(counts) else 0
        weight = weights[i] if i < len(weights) else 0.0
        print(f"{class_name:<20} | {count:<10} | {weight:.4f}")
        
    print("-" * 55 + "\n")
    # ---------------------------------
    
    return torch.tensor(weights, dtype=torch.float), classes

def plot_results(y_true, y_pred, class_names, output_dir):
    """Menyimpan classification report dan confusion matrix ke file."""
    
    # 1. Print & Save Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 2. Plot & Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"\nConfusion Matrix disimpan di: {save_path}")
    plt.close() # Tutup plot agar tidak memakan memori/display error di server