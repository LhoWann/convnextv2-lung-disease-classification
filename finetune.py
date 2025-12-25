import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchvision import datasets, transforms
from model import LungDiseaseModel
from utils import LungDataModule, get_class_weights, plot_results

class CoolDownDataModule(LungDataModule):
    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(p=0.5),
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output_finetune')
    parser.add_argument('--model_name', type=str, default='facebook/convnextv2-tiny-22k-224')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(42)
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    weights_tensor, class_names = get_class_weights(args.data_dir)
    
    data_module = CoolDownDataModule(
        data_dir=args.data_dir, 
        model_name=args.model_name, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    model = LungDiseaseModel.load_from_checkpoint(
        args.checkpoint_path,
        model_name=args.model_name,
        num_classes=len(class_names),
        class_weights=weights_tensor
    )
    
    model.hparams.lr = args.lr

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        monitor='val_f1',
        mode='max',
        filename='finetuned-best',
        save_top_k=1,
        verbose=True
    )

    strategy = 'ddp' if args.devices > 1 else 'auto'
    csv_logger = CSVLogger(save_dir=args.output_dir, name="logs_finetune")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices,
        strategy=strategy,
        precision='16-mixed',
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        default_root_dir=args.output_dir
    )

    trainer.fit(model, data_module)

    if trainer.global_rank == 0:
        best_model_path = checkpoint_callback.best_model_path
        
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        model = LungDiseaseModel.load_from_checkpoint(
            best_model_path,
            class_weights=weights_tensor
        )
        model.eval()
        model.to(device_str)

        y_true = []
        y_pred = []

        data_module.setup()
        test_loader = data_module.test_dataloader()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device_str)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        plot_results(y_true, y_pred, class_names, args.output_dir)