import os
import torch
import pytorch_lightning as pl
from model import LungDiseaseModel
from utils import LungDataModule, get_class_weights, plot_results

if __name__ == '__main__':
    pl.seed_everything(42)
    
    # Konfigurasi
    DATA_DIR = "/kaggle/input/combined-unknown-pneumonia-and-tuberculosis/data"
    MODEL_NAME = "facebook/convnextv2-tiny-22k-224"
    BATCH_SIZE = 128 
    NUM_WORKERS = 4
    OUTPUT_DIR = "./output"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Hitung Weights & Setup Data
    print("Computing class weights...")
    weights_tensor, class_names = get_class_weights(DATA_DIR)
    print(f"Classes: {class_names}")
    print(f"Weights: {weights_tensor}")

    data_module = LungDataModule(DATA_DIR, MODEL_NAME, BATCH_SIZE, NUM_WORKERS)

    # 3. Inisialisasi Model
    model = LungDiseaseModel(
        model_name=MODEL_NAME, 
        num_classes=len(class_names), 
        class_weights=weights_tensor,
        lr=5e-5
    )

    # 4. Setup Training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        monitor='val_f1',
        mode='max',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices=2,              # Menggunakan 2 GPU
        strategy='ddp',         # Distributed Data Parallel
        precision='16-mixed',   # Mixed Precision
        sync_batchnorm=True,    # Wajib untuk DDP di Computer Vision
        callbacks=[checkpoint_callback],
        logger=True,
        default_root_dir=OUTPUT_DIR
    )

    # 5. Jalankan Training
    trainer.fit(model, data_module)

    # 6. Evaluasi (Hanya di proses utama/Rank 0)
    if trainer.global_rank == 0:
        print("\n--- Starting Evaluation ---")
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"Loading best model from: {best_model_path}")
        
        model = LungDiseaseModel.load_from_checkpoint(
            best_model_path,
            class_weights=weights_tensor 
        )
        model.eval()
        model.to('cuda')

        y_true = []
        y_pred = []

        data_module.setup()
        test_loader = data_module.test_dataloader()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to('cuda')
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        plot_results(y_true, y_pred, class_names)