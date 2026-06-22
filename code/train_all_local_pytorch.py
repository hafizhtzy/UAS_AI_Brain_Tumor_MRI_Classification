import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =========================
# KONFIGURASI UTAMA
# =========================
PROJECT_DIR = Path(r"C:\UAS_AI_MRI")
DATA_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"
VIS_DIR = PROJECT_DIR / "visualizations"

for d in [MODEL_DIR, RESULT_DIR, VIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 0  # kalau error di Windows, ubah jadi 0


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_tfms, eval_tfms


def build_dataloaders():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Folder Training tidak ditemukan: {TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Folder Testing tidak ditemukan: {TEST_DIR}")

    train_tfms, eval_tfms = get_transforms()
    train_full_for_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    train_full_for_val = datasets.ImageFolder(TRAIN_DIR, transform=eval_tfms)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)

    class_names = train_full_for_train.classes
    total_size = len(train_full_for_train)
    val_size = int(total_size * VALIDATION_SPLIT)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset_temp = random_split(train_full_for_train, [train_size, val_size], generator=generator)
    train_dataset = Subset(train_full_for_train, train_subset.indices)
    val_dataset = Subset(train_full_for_val, val_subset_temp.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    dataset_info = {
        "train_total_original": total_size,
        "train_used": train_size,
        "validation_used": val_size,
        "test_used": len(test_dataset),
        "classes": class_names,
        "class_to_idx": train_full_for_train.class_to_idx,
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
    }

    with open(RESULT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)

    print("\n=== DATASET INFO ===")
    print(json.dumps(dataset_info, indent=4, ensure_ascii=False))
    return train_loader, val_loader, test_loader, class_names


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

    return model


def run_one_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None):
    model.train() if train else model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def train_model(model_name, train_loader, val_loader, test_loader, class_names, device):
    print(f"\n==============================")
    print(f"TRAINING MODEL: {model_name}")
    print(f"==============================")

    num_classes = len(class_names)
    model = build_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_model_path = MODEL_DIR / f"{model_name}_best.pt"
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_name": model_name, "model_state_dict": model.state_dict(), "class_names": class_names, "img_size": IMG_SIZE, "val_acc": best_val_acc}, best_model_path)

    elapsed = time.time() - start_time
    print(f"Training {model_name} selesai dalam {elapsed/60:.2f} menit.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(RESULT_DIR / f"history_{model_name}.csv", index=False)

    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], marker="o", label="Training Accuracy")
    plt.plot(history["epoch"], history["val_acc"], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"accuracy_curve_{model_name}.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Training Loss")
    plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"loss_curve_{model_name}.png", dpi=300)
    plt.close()

    checkpoint = torch.load(best_model_path, map_location=device)
    model = build_model(model_name, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    test_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_path = RESULT_DIR / f"classification_report_{model_name}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"confusion_matrix_{model_name}.png", dpi=300)
    plt.close()

    result = {
        "Model": model_name,
        "Best Validation Accuracy": round(best_val_acc, 4),
        "Test Accuracy": round(test_accuracy, 4),
        "Precision": round(test_precision, 4),
        "Recall": round(test_recall, 4),
        "F1-Score": round(test_f1, 4),
        "Training Time (minutes)": round(elapsed / 60, 2),
        "Model File": str(best_model_path),
        "Classification Report": str(report_path),
    }
    print("\n=== TEST RESULT ===")
    print(json.dumps(result, indent=4, ensure_ascii=False))
    return result


def save_comparison_graph(comparison_df):
    metrics = ["Test Accuracy", "Precision", "Recall", "F1-Score"]
    for metric in metrics:
        plt.figure()
        plt.bar(comparison_df["Model"], comparison_df[metric])
        plt.ylim(0, 1)
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"Comparison of {metric}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        safe_metric = metric.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(VIS_DIR / f"comparison_{safe_metric}.png", dpi=300)
        plt.close()


def main():
    set_seed(SEED)
    print("Cek CUDA...")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, class_names = build_dataloaders()
    model_list = ["resnet50", "efficientnetb0", "mobilenetv2"]
    all_results = []

    for model_name in model_list:
        result = train_model(model_name, train_loader, val_loader, test_loader, class_names, device)
        all_results.append(result)

    comparison_df = pd.DataFrame(all_results)
    comparison_df.to_csv(RESULT_DIR / "comparison_result.csv", index=False)
    comparison_df.to_excel(RESULT_DIR / "comparison_result.xlsx", index=False)
    save_comparison_graph(comparison_df)

    print("\n==============================")
    print("SEMUA TRAINING SELESAI")
    print("==============================")
    print(comparison_df)
    print(f"\nHasil tersimpan di:")
    print(f"- Model: {MODEL_DIR}")
    print(f"- Result: {RESULT_DIR}")
    print(f"- Visualisasi: {VIS_DIR}")


if __name__ == "__main__":
    main()
