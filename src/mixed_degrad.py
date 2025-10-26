import os, shutil
from tqdm import tqdm

# ---- podesavanja ----
DATA_DIR = "/Users/macbookair/Code/pneumonia-xray-pytorch/data"
ORIG      = os.path.join(DATA_DIR, "chest_xray")
DEGRAD    = {
    "blur":     os.path.join(DATA_DIR, "chest_xray_blur"),
    "contrast": os.path.join(DATA_DIR, "chest_xray_contrast"),
    "noise":    os.path.join(DATA_DIR, "chest_xray_noise"),
    "lowres":   os.path.join(DATA_DIR, "chest_xray_lowres"),
}
DEST      = os.path.join(DATA_DIR, "chest_xray_mixed")

SUBSETS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
# fiksni redosled “četvrtina” -> mapira se po indeksima
ORDER = ["blur", "contrast", "noise", "lowres"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

def base_no_ext(fname):
    return os.path.splitext(fname)[0]

#Function for making the mixed DB
def make_mixed():
    # kreiraj strukturu
    for subset in ["train", "val", "test"]:  # ≤— uključen i test
        for cls in CLASSES:
            ensure_dir(os.path.join(DEST, subset, cls))

    for subset in ["train", "val", "test"]:
        for cls in CLASSES:
            orig_dir = os.path.join(ORIG, subset, cls)
            files = sorted(list_images(orig_dir))
            n = len(files)
            if n == 0:
                continue

            # indeksi četvrtina
            q = [0, round(n/4), round(2*n/4), round(3*n/4), n]

            print(f"[{subset.upper()}] {cls}: {n} fajlova, kvartili={q}")

            for k, mode in enumerate(ORDER):  # ORDER = ["blur","contrast","noise","lowres"]
                start, end = q[k], q[k+1]
                part = files[start:end]
                src_root = DEGRAD[mode]
                dst_dir  = os.path.join(DEST, subset, cls)

                print(f"  - {mode}: {len(part)} fajlova")
                for f in tqdm(part):
                    base = os.path.splitext(f)[0]
                    # uzmi baš postojeći degradirani fajl (isti “ID” slike + sufiks moda)
                    src_path = os.path.join(src_root, subset, cls, f"{base}_{mode}.jpg")
                    if not os.path.exists(src_path):
                        raise FileNotFoundError(src_path)
                    # ❶ zadrži isto ime (bez “mix-”)
                    dst_path = os.path.join(dst_dir, f"{base}_{mode}.jpg")
                    shutil.copy2(src_path, dst_path)

    print("\n✅ Gotov mixed skup (train/val/test disjunktni po slici, bez duplikata):", DEST)

if __name__ == "__main__":
    make_mixed()