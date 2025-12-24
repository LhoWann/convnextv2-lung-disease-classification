import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger 
import warnings
from model import LungDiseaseModel
from utils import LungDataModule, get_class_weights, plot_results, plot_history 

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--model_name', type=str, default='facebook/convnextv2-tiny-22k-224')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--devices', type=int, default=1, help="Jumlah GPU") 
    
    return parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(42)
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available() and torch.cuda.current_device() == 0:
        print(f"Loading data from: {args.data_dir}")

    weights_tensor, class_names = get_class_weights(args.data_dir)
    
    data_module = LungDataModule(
        data_dir=args.data_dir, 
        model_name=args.model_name, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    model = LungDiseaseModel(
        model_name=args.model_name, 
        num_classes=len(class_names), 
        class_weights=weights_tensor,
        lr=args.lr
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        monitor='val_f1',
        mode='max',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True
    )

    strategy = 'ddp' if args.devices > 1 else 'auto'
    sync_bn = True if args.devices > 1 else False

    csv_logger = CSVLogger(save_dir=args.output_dir, name="logs")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices,      
        strategy=strategy,         
        precision='16-mixed',      
        sync_batchnorm=sync_bn,    
        callbacks=[checkpoint_callback],
        logger=csv_logger, 
        default_root_dir=args.output_dir
    )

    trainer.fit(model, data_module)

    plot_history(csv_logger.log_dir, args.output_dir)

    if trainer.global_rank == 0:
        print("\nEvaluation (Rank 0)")
        best_model_path = checkpoint_callback.best_model_path
        print(f"Loading best model from: {best_model_path}")
        
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