from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import torch
from models import PneumoniaClassifier
from PIL import Image
import os
import matplotlib.pyplot as plt
from src import transforms as T
from src import datasets
import pandas as pd
from datetime import datetime
import numpy as np


# --------- Single-image preprocessing ---------
def process_image(image_path, transform):
    """Open an image, ensure RGB, apply test/val transform and add batch dim."""
    image = Image.open(image_path).convert("RGB")
    # transform returns tensor CxHxW; unsqueeze(0) makes it 1xCxHxW (batch of 1)
    return image, transform(image).unsqueeze(0)


# --------- Single-image prediction ---------
def predict(model, image_tensor, device):
    """Run forward pass and return softmax probabilities as a flat numpy array."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy().flatten()


# --------- Simple visual summary for 2-class setup ---------
def visualize_predictions(original_image, probs, class_names):
    """Show the input X-ray and a textual summary of class probabilities."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # left: image
    ax[0].imshow(original_image)
    ax[0].axis("off")
    ax[0].set_title("Input X-ray")

    # right: textual prediction
    pred_idx = probs.argmax()
    pred_class = class_names[pred_idx]
    pred_prob = probs[pred_idx]

    lines = [f"{c}: {p * 100:.2f}%" for c, p in zip(class_names, probs)]

    ax[1].axis("off")
    ax[1].text(0.1, 0.8, f"Predicted: {pred_class}", fontsize=16, fontweight="bold")
    ax[1].text(0.1, 0.6, f"Confidence: {pred_prob * 100:.2f}%", fontsize=14)
    ax[1].text(0.1, 0.4, "\n".join(lines), fontsize=12)

    plt.tight_layout()
    plt.show()


# --------- Full test-set evaluation ---------
def evaluate_model(model, loader, device, class_names):
    """Compute accuracy/precision/recall/F1 on the whole loader and plot CM."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")

    # confusion matrix (quick visual)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Test Set")
    plt.show()

    return acc, prec, rec, f1, all_labels, all_preds


# --------- Save figures + metrics to disk ---------
def save_metrics(
    y_true,
    y_pred,
    class_names,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    run_tag: str = None,
    figs_dir: str = "../outputs/figures",
    metrics_dir: str = "../outputs/metrics",
):
    """
    Save confusion matrix heatmap (PNG) and a small metrics report (XLSX).
    Creates output folders if missing. Returns paths as a small dict.
    """
    # prepare output dirs
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # default suffix for file names
    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    fig_path = os.path.join(figs_dir, f"confusion_matrix_{run_tag}.png")
    xlsx_path = os.path.join(metrics_dir, f"metrics_{run_tag}.xlsx")

    # build confusion matrix figure
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    _ = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", color="black", fontsize=10)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # write XLSX (summary + CM sheet)
    summary = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1"],
        "value":  [accuracy,    precision,   recall,   f1],
    })
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # requires openpyxl in your env
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="summary")
        cm_df.to_excel(writer, sheet_name="confusion_matrix")

    return {"figure_path": fig_path, "excel_path": xlsx_path}


# --------- Load best checkpoint and run a quick test ---------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# rebuild same architecture (weights will be loaded next)
model = PneumoniaClassifier(num_classes=2, freeze_backbone=False)
ckpt = torch.load("../checkpoints/best_model.pt", map_location=device)

# load weights
model.load_state_dict(ckpt["model_state"])
model.to(device)

# quick single-image check (optional)
test_image = "../data/chest_xray/test/NORMAL/IM-0006-0001.jpeg"
original_image, image_tensor = process_image(test_image, transform=T.test_val_transforms)
probs = predict(model, image_tensor, device)

# class names from dataset (or define manually: ['NORMAL', 'PNEUMONIA'])
class_names = datasets.test_dataset.classes

# full test-set evaluation
acc, prec, rec, f1, y_true, y_pred = evaluate_model(
    model=model,
    loader=datasets.test_loader,
    device=device,
    class_names=class_names
)

# save figure + metrics files
paths = save_metrics(y_true, y_pred, class_names, acc, prec, rec, f1)
print("Saved:", paths["figure_path"], "and", paths["excel_path"])

