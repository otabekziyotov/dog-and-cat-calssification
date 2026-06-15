import numpy as np
import streamlit as st
import torch
import timm
from PIL import Image
from pathlib import Path

from transform import get_tfs
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


# ----------------------- CONFIG -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME   = "rexnet_150"
SAVE_PREFIX  = "cat_dog"
WEIGHTS      = PROJECT_ROOT / "saved_models" / f"{SAVE_PREFIX}_best_model.pth"
CLASSES      = ["cat", "dog"]            # index 0 = cat, 1 = dog
IM_SIZE      = 224
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR     = PROJECT_ROOT / "data" / "data_samples" 
N_SAMPLES    = 4                          # how many sample thumbnails per class
# ------------------------------------------------------


@st.cache_resource
def load_model():
    """Build the model and load the trained weights (cached across reruns)."""
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()


def list_samples(class_folder, n):
    """Return up to n sample image paths from the test set for a given class."""
    folder = TEST_DIR / class_folder
    if not folder.exists():
        return []
    return sorted(folder.glob("*"))[:n]


def predict(model, image):
    """Run inference and return probabilities tensor of shape [num_classes]."""
    x = get_tfs(IM_SIZE, MEAN, STD)(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return x, probs


def gradcam_overlay(model, input_tensor, image):
    """Generate a Grad-CAM heatmap overlaid on the original image."""
    cam = GradCAMPlusPlus(model=model, target_layers=[model.features[-1].conv])
    grayscale = cam(input_tensor=input_tensor)[0]
    rgb = np.array(image.resize((IM_SIZE, IM_SIZE))).astype(np.float32) / 255.0
    return show_cam_on_image(rgb, grayscale, use_rgb=True)


# --------------------------- UI ---------------------------
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")
st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image and the model will predict whether it's a **cat** or a **dog**.")

if not WEIGHTS.exists():
    st.error(f"Trained weights not found at `{WEIGHTS}`.\n\n"
             "Run `python main.py` first to train and save the model.")
    st.stop()

model = load_model()

if "sample_path" not in st.session_state:
    st.session_state.sample_path = None

# ----- Sidebar: options + sample images from the test set -----
st.sidebar.header("⚙️ Options")
show_cam = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)

st.sidebar.header("🖼️ Test set samples")
st.sidebar.caption("Click a sample to run it through the model.")
for cls_folder, emoji in [("cat", "🐱"), ("dog", "🐶")]:
    samples = list_samples(cls_folder, N_SAMPLES)
    with st.sidebar.expander(f"{emoji} {cls_folder.capitalize()}s ({len(samples)})", expanded=False):
        if not samples:
            st.write("No samples found. Run `python data/data_downloading.py`.")
        for p in samples:
            st.image(str(p), use_container_width=True)
            if st.button(f"Use this {cls_folder}", key=str(p)):
                st.session_state.sample_path = str(p)

# ----- Main: pick input (uploaded file OR selected sample) -----
file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

image, source = None, ""
if file is not None:
    image = Image.open(file).convert("RGB")
    source = "Uploaded image"
    st.session_state.sample_path = None          # an upload overrides the sample
elif st.session_state.sample_path:
    image = Image.open(st.session_state.sample_path).convert("RGB")
    source = f"Sample: {Path(st.session_state.sample_path).name}"

if image is not None:
    input_tensor, probs = predict(model, image)
    pred = int(torch.argmax(probs))

    st.subheader(f"Prediction: {CLASSES[pred].upper()}  —  {probs[pred] * 100:.1f}% confident")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=source, use_container_width=True)
    with col2:
        if show_cam:
            overlay = gradcam_overlay(model, input_tensor, image)
            st.image(overlay, caption="Grad-CAM (where the model looks)", use_container_width=True)

    st.bar_chart({CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))})
else:
    st.info("⬆️ Upload an image or pick a sample from the sidebar.")
