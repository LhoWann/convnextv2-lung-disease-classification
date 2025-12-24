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
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
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
    """Menghitung class weights untuk imbalanced dataset."""
    train_dir = os.path.join(data_dir, 'train')
    dataset = datasets.ImageFolder(train_dir)
    targets = dataset.targets
    classes = dataset.classes
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    return torch.tensor(weights, dtype=torch.float), classes

def plot_results(y_true, y_pred, class_names):
    """Menampilkan classification report dan confusion matrix."""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()