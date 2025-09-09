# Streamly.auto.py
import os
import io
import glob
import numpy as np
from PIL import Image
import streamlit as st
import torch
import matplotlib.pyplot as plt

# ---- local imports from project ----
from src.models import PneumoniaClassifier
from src.gradcam import GradCAM
from src.transforms import test_val_transforms as T_testval

# ---------- small utils ----------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@st.cache_resource
def load_model_and_device(ckpt_path="checkpoints/best_model.pt"):
    device = get_device()
    model = PneumoniaClassifier(num_classes=2, freeze_backbone=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, device

def process_image_pil(pil_img):
    """Returns (original PIL, tensor [1,3,224,224])."""
    img = pil_img.convert("RGB")
    return img, T_testval(img).unsqueeze(0)

def predict_probs(model, image_tensor, device):
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    return probs  # e.g. [p_normal, p_pneumonia]

def make_overlay(original_pil, heatmap_01, alpha=0.45, cmap="jet"):
    """Blend heatmap [0..1] over original image, return PIL.Image."""
    W, H = original_pil.size
    # resize heatmap to original
    heatmap_resized = np.array(
        Image.fromarray((heatmap_01 * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
    ) / 255.0
    cm = plt.get_cmap(cmap)
    heatmap_rgb = cm(heatmap_resized)[..., :3]  # drop alpha
    img_rgb = np.array(original_pil).astype(np.float32) / 255.0
    overlay = (1 - alpha) * img_rgb + alpha * heatmap_rgb
    overlay = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay)

def list_demo_images():
    demo_dir = "assets/demo_images"
    patterns = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(demo_dir, p)))
    files.sort()
    return files

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Pneumonia X-ray Classifier", layout="wide")
st.title("Pneumonia X-ray Classifier (ResNet18 + Grad-CAM)")

# Sidebar: model checkpoint + mode
st.sidebar.header("Settings")
ckpt_path = st.sidebar.text_input("Path to .pt model", value="checkpoints/best_model.pt")
mode = st.sidebar.radio("Input source", ["Upload image", "Demo image"])

# Try loading model
model = None
device = None
error_load = None
try:
    model, device = load_model_and_device(ckpt_path)
except Exception as e:
    error_load = str(e)

if error_load:
    st.error(f"Could not load model from '{ckpt_path}'.\n{error_load}")
    st.stop()

# Class names (keep it simple)
class_names = ["NORMAL", "PNEUMONIA"]

# Input selection
uploaded_pil = None
selected_path = None

if mode == "Upload image":
    up = st.file_uploader("Upload a chest X-ray (jpg/png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        uploaded_pil = Image.open(io.BytesIO(up.read()))
else:
    demo_files = list_demo_images()
    if not demo_files:
        st.warning("No demo images found in assets/demo_images/")
    else:
        selected_path = st.selectbox("Choose a demo image", demo_files)
        if selected_path:
            uploaded_pil = Image.open(selected_path)

# Run button
run = st.button("Run prediction & Grad-CAM", type="primary")

# Main logic
if run:
    if uploaded_pil is None:
        st.warning("Please upload or choose an image first.")
        st.stop()

    # Preprocess
    original_pil, input_tensor = process_image_pil(uploaded_pil)

    # Prediction
    probs = predict_probs(model, input_tensor, device)
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    pred_prob = float(probs[pred_idx])

    # Grad-CAM
    target_layer = model.baseModel.layer4[-1]  # for timm resnet18
    cam = GradCAM(model, target_layer)
    heatmap_01, _ = cam(input_tensor.to(device), target_index=None)
    overlay_pil = make_overlay(original_pil, heatmap_01, alpha=0.45, cmap="jet")
    cam.remove_hooks()

    # Layout: two columns for images
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Original")
        st.image(original_pil, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM Overlay")
        st.image(overlay_pil, use_container_width=True)
    st.markdown("###  Prediction Results")
    import pandas as pd

    probs_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probs
    })
    probs_df["Confidence (%)"] = (probs_df["Probability"] * 100).round(2)

    colL, colR = st.columns([1, 1], gap="large")

    with colL:
        st.subheader("Prediction")
        st.markdown(f"**Predicted class:** :blue[{pred_name}]")
        st.metric("Confidence", f"{pred_prob * 100:.2f}%")
        # thin progress bar with label
        st.progress(min(max(pred_prob, 0.0), 1.0))

    with colR:
        st.subheader("Class probabilities")
        # bar chart (compact, auto-scaled)
        st.bar_chart(
            probs_df.set_index("Class")["Probability"],
            height=200,
            use_container_width=True,
        )
        # # optional: a neat table under the chart
        # st.dataframe(
        #     probs_df[["Class", "Confidence (%)"]]
        #     .set_index("Class"),
        #     use_container_width=True,
        #     hide_index=False
        # )

    # (Optional) Save overlay to outputs for reference (won't overwrite due to timestamp)

    # os.makedirs("outputs/figures", exist_ok=True)
    # base = "uploaded" if selected_path is None else os.path.splitext(os.path.basename(selected_path))[0]
    # out_path = os.path.join("outputs/figures", f"gradcam_app_{base}.png")
    # try:
    #     overlay_pil.save(out_path)
    #     st.caption(f"Saved overlay to `{out_path}`")
    # except Exception as e:
    #     st.caption(f"Could not save overlay: {e}")

# Footer
st.markdown("---")
st.caption("Educational demo. Not for clinical use.")