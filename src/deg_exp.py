import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src import datasets, transforms as T
from models import PneumoniaClassifierMobileNet, PneumoniaClassifier
from datetime import datetime

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------
# Helper functions
# ---------------------------------------------
def evaluate_model(model, loader, device):
    """Vraća accuracy, precision, recall, f1 za ceo loader."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, prec, rec, f1


def load_test_loader(dataset_root):
    """Pravi test_loader za dati dataset folder."""
    test_dir = os.path.join(dataset_root, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=T.test_val_transforms)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return loader, test_dataset.classes


# ---------------------------------------------
# Config
# ---------------------------------------------
DATA_ROOT_BASE = "../data"
DATASETS = [
    "chest_xray",
    "chest_xray_blur",
    "chest_xray_noise",
    "chest_xray_contrast",
    "chest_xray_lowres",
]

MODELS = {
    "ResNet18": {
        "cls": PneumoniaClassifier,
        "ckpt": "../checkpoints/best_model.pt",
    },
    "MobileNetV2": {
        "cls": PneumoniaClassifierMobileNet,
        "ckpt": "../checkpoints/best_model_mobilenet.pt",
    },
}

OUTPUT_XLSX = "../outputs/metrics/degradation_results.xlsx"

os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)

# ---------------------------------------------
# Main experiment
# ---------------------------------------------
results = []

for model_name, model_info in MODELS.items():
    print(f"\n=== Testing {model_name} ===")

    model = model_info["cls"](num_classes=2, freeze_backbone=False)
    ckpt = torch.load(model_info["ckpt"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    for dataset_name in DATASETS:
        dataset_root = os.path.join(DATA_ROOT_BASE, dataset_name)
        if not os.path.exists(dataset_root):
            print(f"⚠️ Dataset not found: {dataset_root}")
            continue

        loader, class_names = load_test_loader(dataset_root)
        acc, prec, rec, f1 = evaluate_model(model, loader, device)

        print(f"{dataset_name:25s}  "
              f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

        results.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "dataset": dataset_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

# ---------------------------------------------
# Save the results
# ---------------------------------------------
df = pd.DataFrame(results)
df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")

print("\n✅ Rezultati sačuvani u:", OUTPUT_XLSX)