# eval_on_mixed_test.py
import os, torch, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from datetime import datetime

from src import transforms as T
from models import PneumoniaClassifier, PneumoniaClassifierMobileNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_ROOT_BASE   = "../data"
MIXED_TEST_ROOT  = os.path.join(DATA_ROOT_BASE, "chest_xray_mixed", "test")

CKPT_DIR = "../checkpoints"
OUT_XLSX = "../outputs/metrics/degradation_results_mixedTest.xlsx"
os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)

# samo dva modela trenirana na čistim slikama
MODELS = {
    "ResNet18": {
        "cls": PneumoniaClassifier,
        "ckpt": os.path.join(CKPT_DIR, "ResNet18_chest_xray.pt"),
    },
    "MobileNetV2": {
        "cls": PneumoniaClassifierMobileNet,
        "ckpt": os.path.join(CKPT_DIR, "MobileNetV2_chest_xray.pt"),
    },
}

def mixed_test_loader(batch=32):
    ds = ImageFolder(MIXED_TEST_ROOT, transform=T.test_val_transforms)
    return DataLoader(ds, batch_size=batch, shuffle=False), ds.classes

def evaluate(model, loader):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            all_y.extend(y.cpu().numpy())
            all_p.extend(p.cpu().numpy())
    acc = accuracy_score(all_y, all_p)
    prec = precision_score(all_y, all_p, average="binary")
    rec  = recall_score(all_y, all_p, average="binary")
    f1   = f1_score(all_y, all_p, average="binary")
    return acc, prec, rec, f1

def main():
    # test loader iz MIXED seta
    loader, _ = mixed_test_loader()

    rows = []
    for model_name, info in MODELS.items():
        ckpt_path = info["ckpt"]
        if not os.path.isfile(ckpt_path):
            print(f"⚠️ Nema checkpoint-a: {ckpt_path}")
            continue

        print(f"Test {model_name} (trenirano na clean) -> mixed test")
        model = info["cls"](num_classes=2, freeze_backbone=False).to(device)
        blob = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(blob["model_state"])

        acc, prec, rec, f1 = evaluate(model, loader)

        rows.append({
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model":      model_name,
            "Degradacija": "Mixt",
            "Tačnost":    acc,
            "Preciznost": prec,
            "Odziv":      rec,
            "F1":         f1,
        })

    df = pd.DataFrame(rows)
    df.to_excel(OUT_XLSX, index=False, engine="openpyxl")
    print("\n✓ Sačuvano:", OUT_XLSX)

if __name__ == "__main__":
    main()