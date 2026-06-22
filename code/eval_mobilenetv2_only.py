from pathlib import Path
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
MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_best.pt"
RESULT_DIR = PROJECT_DIR / "results"
VIS_DIR = PROJECT_DIR / "visualizations"

IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_NAME = "mobilenetv2"

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

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
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

report_path = RESULT_DIR / "classification_report_mobilenetv2.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_NAME}\n")
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
plt.title("Confusion Matrix - mobilenetv2")
plt.tight_layout()
plt.savefig(VIS_DIR / "confusion_matrix_mobilenetv2.png", dpi=300)
plt.close()

comparison_csv = RESULT_DIR / "comparison_result.csv"
if comparison_csv.exists():
    df = pd.read_csv(comparison_csv)
    new_row = {
        "Model": MODEL_NAME,
        "Best Validation Accuracy": round(checkpoint.get("val_acc", 0), 4),
        "Test Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Training Time (minutes)": "",
        "Model File": str(MODEL_PATH),
        "Classification Report": str(report_path),
    }
    if "Model" in df.columns and (df["Model"] == MODEL_NAME).any():
        for key, value in new_row.items():
            if key in df.columns:
                df.loc[df["Model"] == MODEL_NAME, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(comparison_csv, index=False)
    df.to_excel(RESULT_DIR / "comparison_result.xlsx", index=False)

print("=== MOBILE NET V2 EVALUATION UPDATED ===")
print(f"Best Validation Accuracy: {checkpoint.get('val_acc', 0):.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Saved report to: {report_path}")
print("Updated confusion matrix and comparison result.")
