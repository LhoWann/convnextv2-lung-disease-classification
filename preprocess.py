import torch
from torchvision import transforms
from transformers import AutoImageProcessor

def get_preprocess(model_name='facebook/convnextv2-tiny-22k-224'):
    processor = AutoImageProcessor.from_pretrained(model_name)
    size = (
        processor.size["shortest_edge"],
        processor.size["shortest_edge"]
    ) if "shortest_edge" in processor.size else (224, 224)
    normalize = transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std
    )
    val_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        normalize
    ])
    def preprocess(image):
        tensor = val_transform(image)
        return tensor.unsqueeze(0)  # Add batch dim
    return preprocess
