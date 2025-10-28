# train_variants.py
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from collections import Counter
from datetime import datetime
from tqdm.auto import tqdm

from src import transforms as T
from models import PneumoniaClassifier, PneumoniaClassifierMobileNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_ROOT_BASE = "../data"
VARIANTS = [       # clean
    "chest_xray_blur",
    "chest_xray_noise",
    "chest_xray_contrast",
    "chest_xray_lowres",
    "chest_xray_mixed",      # tvoj mixt skup
]

MODELS = {
    "ResNet18": PneumoniaClassifier,
    "MobileNetV2": PneumoniaClassifierMobileNet,
}

CKPT_DIR = "../checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def make_loaders(dataset_root, batch=32):
    train_ds = ImageFolder(os.path.join(dataset_root, "train"), transform=T.train_transforms)
    val_ds   = ImageFolder(os.path.join(dataset_root, "val"),   transform=T.test_val_transforms)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)
    return train_ds, val_ds, train_ld, val_ld

def class_weights_from_dataset(ds):
    counts = Counter(ds.targets)
    num_classes = len(ds.classes)
    import torch
    t = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float)
    t = torch.clamp(t, min=1.0)
    w = 1.0 / t
    w = w / w.sum()
    return w

def train_one(model, train_ld, val_ld, weight_vec, max_epochs=15):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_vec.to(device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=2)
    best_val = float("inf")
    stall = 0
    patience = 4
    for epoch in range(max_epochs):
        # --- train ---
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_ld, desc=f"Train {epoch+1}/{max_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_loss = run_loss / len(train_ld.dataset)

        # --- val ---
        model.eval()
        run_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_ld, desc=f"Val {epoch+1}/{max_epochs}"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                run_loss += loss.item() * x.size(0)
        val_loss = run_loss / len(val_ld.dataset)

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            stall = 0
            yield {"event": "best", "val_loss": val_loss, "epoch": epoch+1, "model_state": model.state_dict()}
        else:
            stall += 1
            if stall >= patience:
                yield {"event": "early_stop", "epoch": epoch+1}
                return
    yield {"event": "done"}

def main():
    for variant in VARIANTS:
        root = os.path.join(DATA_ROOT_BASE, variant)
        if not os.path.isdir(root):
            print(f"Skip (not found): {root}")
            continue

        train_ds, val_ds, train_ld, val_ld = make_loaders(root)
        w = class_weights_from_dataset(train_ds)

        for model_name, ModelCls in MODELS.items():
            print(f"\n=== Train {model_name} on {variant} ===")
            model = ModelCls(num_classes=2, freeze_backbone=True)  # partial fine-tuning

            ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_{variant}.pt")
            best_blob = None
            for ev in train_one(model, train_ld, val_ld, w):
                if ev["event"] == "best":
                    best_blob = {
                        "model_state": ev["model_state"],
                        "val_loss": ev["val_loss"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "variant": variant,
                        "model_name": model_name,
                    }
                    torch.save(best_blob, ckpt_path)
                    print(f"  ✓ saved best -> {ckpt_path} (val_loss={ev['val_loss']:.4f})")
                elif ev["event"] == "early_stop":
                    print(f"  (early stop @ epoch {ev['epoch']})")
                    break
            print("Done.")

if __name__ == "__main__":
    main()