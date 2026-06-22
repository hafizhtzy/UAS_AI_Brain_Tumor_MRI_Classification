from pathlib import Path
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

PROJECT_DIR = Path(r"C:\UAS_AI_MRI")
MODEL_DIR = PROJECT_DIR / "models"
IMG_SIZE = 224


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")
    return model


def load_checkpoint(model_name: str, device):
    ckpt_path = MODEL_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(model_name, len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(image_path: str, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(model_name, device)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class = class_names[pred_idx.item()]
    confidence_value = confidence.item()
    print(f"Model          : {model_name}")
    print(f"Image          : {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence     : {confidence_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path gambar MRI yang ingin diprediksi")
    parser.add_argument("--model", default="efficientnetb0", choices=["resnet50", "efficientnetb0", "mobilenetv2"])
    args = parser.parse_args()
    predict_image(args.image, args.model)
