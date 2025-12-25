import os
import torch
import numpy as np
import pandas as pd 
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

class LungDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, model_name, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        self.size = (
            self.processor.size["shortest_edge"],
            self.processor.size["shortest_edge"]
        ) if "shortest_edge" in self.processor.size else (224, 224)
        
        self.normalize = transforms.Normalize(
            mean=self.processor.image_mean, 
            std=self.processor.image_std
        )

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            self.normalize
        ])

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
    train_dir = os.path.join(data_dir, 'train')
    
    dataset = datasets.ImageFolder(train_dir)
    targets = dataset.targets
    classes = dataset.classes
    
    unique, counts = np.unique(targets, return_counts=True)
    total_samples = sum(counts)
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    
    print(f"Total samples: {total_samples}")
    print("-" * 55)
    print(f"{'Class Name':<20} | {'Count':<10} | {'Weight':<10}")
    print("-" * 55)
    
    for i, class_name in enumerate(classes):
        count = counts[i] if i < len(counts) else 0
        weight = weights[i] if i < len(weights) else 0.0
        print(f"{class_name:<20} | {count:<10} | {weight:.4f}")
        
    print("-" * 55 + "\n")
    
    return torch.tensor(weights, dtype=torch.float), classes

# Fungsi untuk plot history training
def plot_history(log_dir, output_dir):
    """Membaca metrics.csv dari CSVLogger dan membuat plot Loss & Accuracy."""
    metrics_path = os.path.join(log_dir, "metrics.csv")
    
    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} tidak ditemukan. Grafik history tidak dibuat.")
        return

    df = pd.read_csv(metrics_path)
    
    # Agregasi per epoch (karena log bisa per step)
    if 'epoch' in df.columns:
        epoch_metrics = df.groupby('epoch').mean()
    else:
        epoch_metrics = df # Fallback

    plt.figure(figsize=(12, 5))
    
    # 1. Plot Loss
    plt.subplot(1, 2, 1)
    if 'train_loss' in epoch_metrics.columns:
        plt.plot(epoch_metrics.index, epoch_metrics['train_loss'], label='Train Loss', marker='.')
    if 'val_loss' in epoch_metrics.columns:
        plt.plot(epoch_metrics.index, epoch_metrics['val_loss'], label='Val Loss', marker='.')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. Plot Accuracy (Mencari kolom yang mengandung 'acc')
    plt.subplot(1, 2, 2)
    train_acc_cols = [c for c in epoch_metrics.columns if 'train' in c and ('acc' in c or 'f1' in c)]
    val_acc_cols = [c for c in epoch_metrics.columns if 'val' in c and ('acc' in c or 'f1' in c)]
    
    for col in train_acc_cols:
        plt.plot(epoch_metrics.index, epoch_metrics[col], label=col, marker='.')
    for col in val_acc_cols:
        plt.plot(epoch_metrics.index, epoch_metrics[col], label=col, marker='.')
        
    plt.title('Training & Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_graph.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Grafik training disimpan di: {save_path}")

def plot_results(y_true, y_pred, class_names, output_dir):
    """Menyimpan classification report dan confusion matrix dengan detail akurasi."""
    
    # 1. Print & Save Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    overall_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    # Akurasi per kelas (Recall) = Diagonal / Total baris
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    detail_text = f"Overall Accuracy: {overall_acc:.2%}\n\nPer-class Accuracy:\n"
    for name, acc in zip(class_names, class_acc):
        detail_text += f"{name}: {acc:.2%}\n"

    # 2. Plot & Save Confusion Matrix
    plt.figure(figsize=(10, 9)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Overall Acc: {overall_acc:.2%})')
    
    # teks detail akurasi di bawah plot
    plt.figtext(0.5, 0.02, detail_text, wrap=True, horizontalalignment='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0.25, 1, 1]) 
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"\nConfusion Matrix Saved to {save_path}")
    plt.close()