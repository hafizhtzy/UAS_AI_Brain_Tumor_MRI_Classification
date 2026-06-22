from pathlib import Path
import argparse
import re
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(r"C:\UAS_AI_MRI")
TEST_DIR = PROJECT_DIR / "dataset" / "Testing"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"
VIS_DIR = PROJECT_DIR / "visualizations"

IMG_SIZE = 224
BATCH_SIZE = 16

def build_model(model_name: str, num_classes: int):
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

def evaluate_model(model_name: str):
    model_path = MODEL_DIR / f"{model_name}_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(model_name, len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_path = RESULT_DIR / f"classification_report_{model_name}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best Validation Accuracy: {checkpoint.get('val_acc', 0):.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test Precision: {prec:.4f}\n")
        f.write(f"Test Recall: {rec:.4f}\n")
        f.write(f"Test F1-Score: {f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"confusion_matrix_{model_name}.png", dpi=300)
    plt.close()

    print(f"\n=== {model_name.upper()} EVALUATION UPDATED ===")
    print(f"Best Validation Accuracy: {checkpoint.get('val_acc', 0):.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Saved report to: {report_path}")

    return {
        "Model": model_name,
        "Best Validation Accuracy": round(float(checkpoint.get("val_acc", 0)), 4),
        "Test Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1-Score": round(float(f1), 4),
        "Model File": str(model_path),
        "Classification Report": str(report_path),
    }

def parse_report(model_name: str):
    path = RESULT_DIR / f"classification_report_{model_name}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    def find_float(label):
        m = re.search(rf"{label}:\s*([0-9.]+)", text)
        return round(float(m.group(1)), 4) if m else None

    return {
        "Model": model_name,
        "Best Validation Accuracy": find_float("Best Validation Accuracy"),
        "Test Accuracy": find_float("Test Accuracy"),
        "Precision": find_float("Test Precision"),
        "Recall": find_float("Test Recall"),
        "F1-Score": find_float("Test F1-Score"),
        "Model File": str(MODEL_DIR / f"{model_name}_best.pt"),
        "Classification Report": str(path),
    }

def rebuild_comparison():
    rows = []
    for model_name in ["resnet50", "efficientnetb0", "mobilenetv2"]:
        row = parse_report(model_name)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "comparison_result_final.csv", index=False)
    df.to_excel(RESULT_DIR / "comparison_result_final.xlsx", index=False)

    if not df.empty:
        metrics = ["Test Accuracy", "Precision", "Recall", "F1-Score"]
        for metric in metrics:
            plt.figure()
            plt.bar(df["Model"], df[metric])
            plt.ylim(0, 1)
            plt.xlabel("Model")
            plt.ylabel(metric)
            plt.title(f"Final Comparison of {metric}")
            plt.xticks(rotation=15)
            plt.tight_layout()
            safe_metric = metric.lower().replace(" ", "_").replace("-", "_")
            plt.savefig(VIS_DIR / f"final_comparison_{safe_metric}.png", dpi=300)
            plt.close()

    print("\n=== FINAL COMPARISON REBUILT ===")
    print(df)
    print(f"Saved to: {RESULT_DIR / 'comparison_result_final.csv'}")
    print(f"Saved to: {RESULT_DIR / 'comparison_result_final.xlsx'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["resnet50", "efficientnetb0", "mobilenetv2", "all"],
        default="all",
        help="Model yang mau dievaluasi ulang. Default: all."
    )
    args = parser.parse_args()

    if args.model == "all":
        for model_name in ["resnet50", "efficientnetb0", "mobilenetv2"]:
            evaluate_model(model_name)
    else:
        evaluate_model(args.model)

    rebuild_comparison()
