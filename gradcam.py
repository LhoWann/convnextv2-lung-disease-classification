import argparse
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor
from model import LungDiseaseModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help="Path ke file gambar input")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path ke file .ckpt (best-checkpoint)")
    parser.add_argument('--model_name', type=str, default='facebook/convnextv2-tiny-22k-224')
    parser.add_argument('--target_class', type=int, default=None, help="Index kelas target (Opsional, 0=Normal, 1=Pneumonia, 2=TBC, 3=Unknown)")
    parser.add_argument('--output', type=str, default='gradcam_result.jpg', help="Nama file output")
    return parser.parse_args()

def run_gradcam():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[INFO] Loading model from {args.checkpoint_path}...")
    
    pl_model = LungDiseaseModel.load_from_checkpoint(
        args.checkpoint_path, 
        model_name=args.model_name, 
        num_classes=4, 
        class_weights=torch.ones(4)
    )
    pl_model.eval()
    pl_model.to(device)
    
    target_layers = [pl_model.model.convnextv2.encoder.stages[-1].layers[-1]]

    print(f"[INFO] Processing image: {args.image_path}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    
    size = (processor.size["shortest_edge"], processor.size["shortest_edge"]) \
           if "shortest_edge" in processor.size else (224, 224)
    
    img_pil = Image.open(args.image_path).convert("RGB")
    img_pil_resized = img_pil.resize(size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    
    input_tensor = transform(img_pil_resized).unsqueeze(0).to(device)
    
    rgb_img = np.float32(img_pil_resized) / 255
    
    cam = GradCAM(model=pl_model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(args.target_class)] if args.target_class is not None else None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    cv2.imwrite(args.output, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"[SUCCESS] Grad-CAM disimpan di: {args.output}")
    
    with torch.no_grad():
        logits = pl_model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        
        classes = ['Normal', 'Pneumonia', 'Tuberculosis', 'Unknown'] 
        
        print(f"\n--- Hasil Prediksi ---")
        print(f"Kelas Terprediksi: {classes[pred_idx]}")
        print(f"Confidence: {probs[0][pred_idx]:.2%}")
        
        if args.target_class is not None:
             print(f"Grad-CAM Target: {classes[args.target_class]}")
        else:
             print(f"Grad-CAM Target: {classes[pred_idx]} (Default: Top Prediction)")

if __name__ == '__main__':
    run_gradcam()