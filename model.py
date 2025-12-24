import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import ConvNextV2ForImageClassification
from torchmetrics import Accuracy, F1Score

class LungDiseaseModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, class_weights, lr=5e-5, weight_decay=0.05):
        super().__init__()
        self.save_hyperparameters()
        
        # Pretrained Model
        self.model = ConvNextV2ForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Loss Function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        
        # sync_dist=True penting untuk DDP logging
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', self.train_acc, prog_bar=True, sync_dist=True)
        self.log('train_f1', self.train_f1, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, sync_dist=True)
        self.log('val_f1', self.val_f1, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]