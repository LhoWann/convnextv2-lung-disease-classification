import os
import argparse
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import LungDataModule, get_class_weights
from model import LungDiseaseModel

# Fungsi untuk mengembalikan warna gambar (Un-normalize)
# Agar gambar tidak terlihat 'aneh' atau terlalu kontras saat ditampilkan
def denormalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) # Reverse: x * std + mean
    return tensor

def visualize_preprocessing(args):
    """
    Membandingkan gambar asli (Raw) dengan gambar setelah Augmentasi.
    """
    
    # Init DataModule untuk meminjam transformasinya
    dm = LungDataModule(args.data_dir, args.model_name)
    dm.setup()
    
    # Ambil daftar file dari folder train
    train_dir = os.path.join(args.data_dir, 'train')
    classes = sorted(os.listdir(train_dir))
    
    # Ambil 1 contoh gambar random dari setiap kelas
    sample_files = []
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        files = os.listdir(cls_path)
        if files:
            img_name = random.choice(files)
            sample_files.append((cls, os.path.join(cls_path, img_name)))

    # Plotting
    fig, axes = plt.subplots(len(sample_files), 2, figsize=(8, 4 * len(sample_files)))
    fig.suptitle(f"Before vs After Preprocessing (Augmentation)", fontsize=16)

    for i, (cls_name, img_path) in enumerate(sample_files):
        # 1. Gambar Asli (Before)
        raw_image = Image.open(img_path).convert('RGB')
        
        # 2. Gambar Augmentasi (After)
        # Kita aplikasikan transform yang ada di DataModule
        augmented_tensor = dm.train_ds.transform(raw_image)
        
        # Denormalize agar enak dilihat manusia
        display_tensor = denormalize(augmented_tensor, dm.processor.image_mean, dm.processor.image_std)
        display_image = display_tensor.permute(1, 2, 0).numpy() # C,H,W -> H,W,C
        display_image = np.clip(display_image, 0, 1)

        # Plot Raw
        axes[i, 0].imshow(raw_image)
        axes[i, 0].set_title(f"Original: {cls_name}")
        axes[i, 0].axis('off')

        # Plot Augmented
        axes[i, 1].imshow(display_image)
        axes[i, 1].set_title(f"Processed (Input Model)")
        axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, 'viz_preprocessing.png')
    plt.savefig(save_path)
    print(f"Gambar disimpan di: {save_path}")
    plt.close()

def visualize_predictions(args):
    """
    Menampilkan contoh prediksi BENAR dan SALAH dari model Fine-Tuned.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data & Model
    weights_tensor, class_names = get_class_weights(args.data_dir)
    dm = LungDataModule(args.data_dir, args.model_name, batch_size=16) # Batch kecil cukup
    dm.setup()
    
    model = LungDiseaseModel.load_from_checkpoint(
        args.checkpoint_path,
        model_name=args.model_name,
        num_classes=len(class_names),
        class_weights=weights_tensor
    )
    model.eval()
    model.to(device)
    
    correct_samples = []
    incorrect_samples = []
    
    # Loop Test Set (Cukup ambil beberapa batch sampai kita punya cukup sampel)
    test_loader = dm.test_dataloader()
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            
            # Simpan sampel
            for i in range(len(labels)):
                img_tensor = inputs[i].cpu()
                pred_idx = preds[i].item()
                true_idx = labels[i].item()
                
                # Denormalize untuk display
                img_display = denormalize(img_tensor, dm.processor.image_mean, dm.processor.image_std)
                
                sample_info = {
                    "image": img_display,
                    "pred": class_names[pred_idx],
                    "true": class_names[true_idx]
                }
                
                if pred_idx == true_idx:
                    correct_samples.append(sample_info)
                else:
                    incorrect_samples.append(sample_info)
            
            # Stop jika sudah punya cukup sampel (misal masing-masing 10)
            if len(correct_samples) > 10 and len(incorrect_samples) > 10:
                break
    
    # Plot Hasil Salah (Incorrect)
    if incorrect_samples:
        n_show = min(5, len(incorrect_samples))
        fig, axes = plt.subplots(1, n_show, figsize=(15, 4))
        if n_show == 1: axes = [axes]
        fig.suptitle(f"Model Errors", fontsize=16, color='red')
        
        for i in range(n_show):
            sample = incorrect_samples[i]
            img = sample['image'].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f"True: {sample['true']}\nPred: {sample['pred']}", color='red', fontweight='bold')
            axes[i].axis('off')
            
        save_path = os.path.join(args.output_dir, 'viz_incorrect_preds.png')
        plt.savefig(save_path)
        print(f"Gambar Error disimpan di: {save_path}")
        plt.close()
    else:
        print("Hebat! Tidak ditemukan prediksi salah pada batch awal.")

    # Plot Hasil Benar (Correct)
    if correct_samples:
        n_show = min(5, len(correct_samples))
        fig, axes = plt.subplots(1, n_show, figsize=(15, 4))
        if n_show == 1: axes = [axes]
        fig.suptitle(f"Model Correct Predictions", fontsize=16, color='green')
        
        for i in range(n_show):
            sample = correct_samples[i]
            img = sample['image'].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f"True: {sample['true']}\nPred: {sample['pred']}", color='green')
            axes[i].axis('off')
            
        save_path = os.path.join(args.output_dir, 'viz_correct_preds.png')
        plt.savefig(save_path)
        print(f"Gambar Benar disimpan di: {save_path}")
        plt.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path ke file .ckpt hasil fine tune")
    parser.add_argument('--output_dir', type=str, default='./output_viz')
    parser.add_argument('--model_name', type=str, default='facebook/convnextv2-tiny-22k-224')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Visualisasi Preprocessing (Data Mentah vs Augmentasi)
    visualize_preprocessing(args)
    
    # 2. Visualisasi Prediksi Model (Benar vs Salah)
    visualize_predictions(args)