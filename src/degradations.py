import os
import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------------
# Simple image degratadion
# -----------------------------------
def degrade_image(img, mode="blur"):
    if mode == "blur":
        return cv2.GaussianBlur(img, (5, 5), 1.2)
    elif mode == "noise":
        noise = np.random.normal(0, 15, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)
    elif mode == "contrast":
        return cv2.convertScaleAbs(img, alpha=0.8, beta=20)
    elif mode == "lowres":
        h, w = img.shape[:2]
        img_small = cv2.resize(img, (w//2, h//2))
        return cv2.resize(img_small, (w, h))
    else:
        raise ValueError(f"Unknown degradation mode: {mode}")

# -----------------------------------
#   Main function
# -----------------------------------
def create_degraded_dataset(
    src_root="/Users/macbookair/Code/pneumonia-xray-pytorch/data/chest_xray",
    mode="blur"
):
    """
    Kreira novi dataset u istom formatu kao original:
    - chest_xray_<mode>/
        - train/NORMAL, train/PNEUMONIA
        - val/NORMAL, val/PNEUMONIA
        - test/NORMAL, test/PNEUMONIA
    Svaka slika se degradira i čuva sa sufiksom: _<mode>.jpg
    """
    dest_root = os.path.join(os.path.dirname(src_root), f"chest_xray_{mode}")
    os.makedirs(dest_root, exist_ok=True)

    subsets = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]

    for subset in subsets:
        for cls in classes:
            src_dir = os.path.join(src_root, subset, cls)
            dst_dir = os.path.join(dest_root, subset, cls)
            os.makedirs(dst_dir, exist_ok=True)

            img_files = [f for f in os.listdir(src_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            print(f"[{mode.upper()}] Processing {subset}/{cls}: {len(img_files)} images")

            for fname in tqdm(img_files):
                src_path = os.path.join(src_dir, fname)
                dst_name = os.path.splitext(fname)[0] + f"_{mode}.jpg"
                dst_path = os.path.join(dst_dir, dst_name)

                try:
                    img = cv2.imread(src_path)
                    if img is None:
                        print(f"⚠️ Could not read {src_path}")
                        continue

                    degraded = degrade_image(img, mode)
                    cv2.imwrite(dst_path, degraded)

                except Exception as e:
                    print(f"Error processing {src_path}: {e}")

    print(f"\n✅ Finished creating degraded dataset: chest_xray_{mode}\n")


# -----------------------------------
# Running the script
# -----------------------------------
if __name__ == "__main__":
    create_degraded_dataset(mode="lowres")  # 4️⃣ lowres