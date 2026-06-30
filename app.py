"""
app.py

AResUNet — Attention Residual U-Net for Building Segmentation
Streamlit application.

Run with:
    streamlit run app.py
"""

import os
import io
import time

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from model import AResUNet
from utils import (
    IMG_SIZE,
    THRESHOLD,
    preprocess_mask,
    run_inference,
    dice_score_np,
    iou_score_np,
    accuracy_score_np,
    f1_score_np,
    building_percentage,
    make_overlay,
    make_error_map,
    mask_to_display,
)

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AResUNet | Building Segmentation",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "newmodel333 (2).pth")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")
SAMPLE_MASKS_DIR = os.path.join(BASE_DIR, "sample_masks")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

VALID_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Custom CSS — modern look
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }

    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .app-subtitle {
        font-size: 1.15rem;
        color: #a9b4c0;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        background: #1b1f27;
        border: 1px solid #2a2f3a;
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
    }

    .metric-card h3 {
        font-size: 0.85rem;
        color: #9aa4b2;
        font-weight: 500;
        margin-bottom: 6px;
    }

    .metric-card p {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        color: #e6e9ee;
        border-left: 4px solid #0072ff;
        padding-left: 10px;
    }

    .legend-box {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin-right: 6px;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AResUNet model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AResUNet()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    return model, device, device_name


def list_sample_images():
    if not os.path.isdir(SAMPLE_IMAGES_DIR):
        return []
    return sorted(
        f for f in os.listdir(SAMPLE_IMAGES_DIR)
        if f.lower().endswith(VALID_IMAGE_EXT)
    )


def find_matching_mask(image_filename: str):
    """Look for a ground-truth mask in sample_masks/ with the same base name."""
    if not os.path.isdir(SAMPLE_MASKS_DIR):
        return None

    base_name = os.path.splitext(image_filename)[0]

    for f in os.listdir(SAMPLE_MASKS_DIR):
        if os.path.splitext(f)[0] == base_name and f.lower().endswith(VALID_IMAGE_EXT):
            return os.path.join(SAMPLE_MASKS_DIR, f)

    return None


def read_rgb_image(path_or_bytes, is_path: bool):
    if is_path:
        img = cv2.imread(path_or_bytes)
        if img is None:
            # fall back to PIL for formats cv2 struggles with (e.g. some .tif)
            pil_img = Image.open(path_or_bytes).convert("RGB")
            return np.array(pil_img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        pil_img = Image.open(io.BytesIO(path_or_bytes)).convert("RGB")
        return np.array(pil_img)


def to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏙️ AResUNet")
    st.caption("Attention Residual U-Net")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "ℹ️ About Model", "🖼️ Image Selection"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    model, device, device_name = load_model()

    if model is not None:
        st.success(f"Model loaded ✅\n\nDevice: **{device_name}**")
    else:
        st.error("Model file not found ❌")

    st.markdown("---")
    st.caption("Built with Streamlit · PyTorch")


# ---------------------------------------------------------------------
# Session state for selected image / prediction flow
# ---------------------------------------------------------------------
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None       # np.ndarray (RGB)
if "selected_image_name" not in st.session_state:
    st.session_state.selected_image_name = None
if "selected_image_source" not in st.session_state:
    st.session_state.selected_image_source = None  # "sample" or "upload"
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
if "scroll_to_preview" not in st.session_state:
    st.session_state.scroll_to_preview = False


def reset_selection():
    st.session_state.selected_image = None
    st.session_state.selected_image_name = None
    st.session_state.selected_image_source = None
    st.session_state.show_prediction = False
    st.session_state.scroll_to_preview = False


# ---------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<p class="app-title">AResUNet</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Attention Residual U-Net for Building Segmentation</p>',
        unsafe_allow_html=True,
    )

    st.write(
        "AResUNet is a custom segmentation architecture built on a ResNet-34 "
        "encoder, enhanced with **ASPP**, **Attention Gates**, **SE-augmented "
        "residual decoders**, a **Boundary Refinement Block**, **Multi-Scale "
        "Feature Fusion (MSFF)**, and **CBAM** attention — designed to extract "
        "buildings from satellite / aerial imagery with sharp, accurate boundaries."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><h3>Input Resolution</h3>'
            f'<p>{IMG_SIZE} × {IMG_SIZE}</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><h3>Decision Threshold</h3>'
            f'<p>{THRESHOLD}</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><h3>Device</h3>'
            f'<p>{device_name if device_name else "N/A"}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-header">Get Started</p>', unsafe_allow_html=True)
    st.write(
        "Use the sidebar to explore the model architecture, or jump to "
        "**Image Selection** to choose a sample image or upload your own and "
        "run building segmentation."
    )

# ---------------------------------------------------------------------
# PAGE: About Model
# ---------------------------------------------------------------------
elif page == "ℹ️ About Model":
    st.markdown('<p class="section-header">About AResUNet</p>', unsafe_allow_html=True)

    st.write(
        "AResUNet (Attention Residual U-Net) extends a standard U-Net with "
        "several enhancements aimed at improving building boundary accuracy "
        "in satellite imagery segmentation:"
    )

    st.markdown(
        """
- **Encoder:** ResNet-34 (via `segmentation_models_pytorch`)
- **ASPP (Atrous Spatial Pyramid Pooling):** captures multi-scale context at the bottleneck
- **Attention Gates:** suppress irrelevant skip-connection activations before decoding
- **Decoder Blocks:** residual decoder blocks with dilated convolutions and Squeeze-and-Excitation (SE) attention
- **Boundary Refinement Block:** dilated residual block to sharpen building edges
- **MSFF (Multi-Scale Feature Fusion):** fuses multi-receptive-field features at the final decoder stage
- **CBAM:** channel + spatial attention applied before the segmentation head
- **Deep Supervision:** auxiliary heads on intermediate decoder stages during training (disabled at inference)
        """
    )

    st.markdown('<p class="section-header">Training Configuration</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
- **Input size:** {IMG_SIZE} × {IMG_SIZE}, RGB, normalized to [0, 1]
- **Loss function:** weighted combination of BCE (0.3), Dice (0.5), and Focal Loss (0.2)
- **Optimizer:** AdamW (lr = 1e-4, weight decay = 1e-4)
- **Prediction threshold:** {THRESHOLD}
- **Evaluation metrics:** Dice Score, IoU, Pixel Accuracy, F1 Score
        """
    )

    if model is not None:
        n_params = sum(p.numel() for p in model.parameters())
        st.markdown('<p class="section-header">Loaded Checkpoint</p>', unsafe_allow_html=True)
        st.write(f"**Parameters:** {n_params:,}")
        st.write(f"**Checkpoint file:** `newmodel333 (2).pth`")
        st.write(f"**Running on:** {device_name}")
    else:
        st.warning(
            "Model checkpoint not found. Place `newmodel333 (2).pth` inside "
            "the `model/` folder to load architecture details here."
        )

# ---------------------------------------------------------------------
# PAGE: Image Selection (Sample Images + Upload Image + Prediction flow)
# ---------------------------------------------------------------------
elif page == "🖼️ Image Selection":

    # ===================================================================
    # PREDICTION SCREEN (shown after "Predict Now" is clicked)
    # ===================================================================
    if st.session_state.show_prediction:

        st.markdown('<p class="section-header">Run Prediction</p>', unsafe_allow_html=True)

        if model is None:
            st.error(
                "⚠️ Model checkpoint not found at `model/newmodel333 (2).pth`. "
                "Please copy your trained model file into the `model/` folder and "
                "restart the app."
            )
            st.stop()

        if st.session_state.selected_image is None:
            st.info(
                "No image selected yet. Please go back and choose a sample "
                "image or upload one first."
            )
            st.stop()

        image_rgb = st.session_state.selected_image
        image_name = st.session_state.selected_image_name
        source = st.session_state.selected_image_source

        st.write(f"**Selected image:** `{image_name}`  ·  **Source:** {source}")

        with st.spinner("Running inference..."):
            binary_mask, inference_time = run_inference(model, image_rgb, device)

        # Resize original image to match mask resolution for display/overlay
        image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
        mask_display = mask_to_display(binary_mask)
        overlay = make_overlay(image_resized, binary_mask)

        # -------------------- Display: Image / Mask / Overlay --------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original Image**")
            st.image(image_resized, use_container_width=True)
        with col2:
            st.markdown("**Predicted Mask**")
            st.image(mask_display, use_container_width=True, clamp=True)
        with col3:
            st.markdown("**Overlay (Green = Building)**")
            st.image(overlay, use_container_width=True)

        # -------------------- Stats --------------------
        st.markdown('<p class="section-header">Inference Details</p>', unsafe_allow_html=True)

        bperc = building_percentage(binary_mask)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card"><h3>Inference Time</h3><p>{inference_time*1000:.1f} ms</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><h3>Device Used</h3><p>{device_name}</p></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><h3>Image Resolution</h3><p>{IMG_SIZE} × {IMG_SIZE}</p></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-card"><h3>Building %</h3><p>{bperc:.2f}%</p></div>',
                unsafe_allow_html=True,
            )

        # -------------------- Ground truth comparison --------------------
        gt_path = None
        if source == "sample":
            gt_path = find_matching_mask(image_name)

        if gt_path is not None:
            st.markdown('<p class="section-header">Ground Truth Comparison</p>', unsafe_allow_html=True)

            gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = preprocess_mask(gt_raw)

            dice = dice_score_np(binary_mask, gt_mask)
            iou = iou_score_np(binary_mask, gt_mask)
            acc = accuracy_score_np(binary_mask, gt_mask)
            f1 = f1_score_np(binary_mask, gt_mask)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><h3>Dice Score</h3><p>{dice:.4f}</p></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><h3>IoU</h3><p>{iou:.4f}</p></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><h3>Pixel Accuracy</h3><p>{acc:.4f}</p></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-card"><h3>F1 Score</h3><p>{f1:.4f}</p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<p class="section-header">Ground Truth Comparison</p>', unsafe_allow_html=True)

            error_map = make_error_map(gt_mask, binary_mask)

            e1, e2, e3 = st.columns(3)

            with e1:
                st.markdown("**Ground Truth Mask**")
                st.image(
                    mask_to_display(gt_mask),
                    use_container_width=True,
                    clamp=True

                )

            with e2:
                st.markdown("**Predicted Mask**")
                st.image(
                    mask_display,
                    use_container_width=True,
                    clamp=True
                )

            with e3:
                st.markdown("**Error Map**")
                st.image(
                    error_map,
                    use_container_width=True
                )

            st.markdown(
                """
                <span class="legend-box" style="background:#00ff00;"></span> True Positive &nbsp;&nbsp;
                <span class="legend-box" style="background:#ff0000;"></span> False Positive &nbsp;&nbsp;
                <span class="legend-box" style="background:#0000ff;"></span> False Negative &nbsp;&nbsp;
                <span class="legend-box" style="background:#000000; border:1px solid #444;"></span> True Negative
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <span class="legend-box" style="background:#00ff00;"></span> True Positive &nbsp;&nbsp;
                <span class="legend-box" style="background:#ff0000;"></span> False Positive &nbsp;&nbsp;
                <span class="legend-box" style="background:#0000ff;"></span> False Negative &nbsp;&nbsp;
                <span class="legend-box" style="background:#000000; border:1px solid #444;"></span> True Negative
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No matching ground-truth mask found in `sample_masks/` for this "
                "image. Metrics and error map are only available for sample "
                "images that have a corresponding mask with the same filename."
            )

        # -------------------- Downloads --------------------
        st.markdown('<p class="section-header">Downloads</p>', unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "⬇️ Download Prediction (Mask)",
                data=to_png_bytes(mask_display),
                file_name=f"prediction_{os.path.splitext(image_name)[0]}.png",
                mime="image/png",
            )
        with d2:
            st.download_button(
                "⬇️ Download Binary Mask",
                data=to_png_bytes((binary_mask * 255).astype(np.uint8)),
                file_name=f"binary_mask_{os.path.splitext(image_name)[0]}.png",
                mime="image/png",
            )
        with d3:
            st.download_button(
                "⬇️ Download Overlay",
                data=to_png_bytes(overlay),
                file_name=f"overlay_{os.path.splitext(image_name)[0]}.png",
                mime="image/png",
            )

        st.markdown("---")
        if st.button("⬅ Select Another Image", use_container_width=True):
            reset_selection()
            st.rerun()

    # ===================================================================
    # IMAGE SELECTION SCREEN (Sample Images + Upload Image)
    # ===================================================================
    else:
        st.markdown('<p class="section-header">Image Selection</p>', unsafe_allow_html=True)

        left_col, right_col = st.columns(2)

        # -------------------- Sample Images --------------------
        with left_col:
            st.markdown("### 📁 Sample Images")
            samples = list_sample_images()

            if not samples:
                st.info(
                    "No sample images found. Copy some images into the "
                    "`sample_images/` folder to see them listed here."
                )
            else:
                sample_cols = st.columns(3)
                for i, fname in enumerate(samples):
                    path = os.path.join(SAMPLE_IMAGES_DIR, fname)
                    with sample_cols[i % 3]:
                        try:
                            img = read_rgb_image(path, is_path=True)
                            st.image(img, caption=fname, use_container_width=True)
                            if st.button("Select", key=f"select_{fname}"):
                                st.session_state.selected_image = img
                                st.session_state.selected_image_name = fname
                                st.session_state.selected_image_source = "sample"
                                st.session_state.show_prediction = False
                                st.session_state.scroll_to_preview = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"Could not load {fname}: {e}")

        # -------------------- Upload Image --------------------
        with right_col:
            st.markdown("### ⬆️ Upload Image")

            uploaded_file = st.file_uploader(
                "Upload a PNG / JPG / JPEG image",
                type=["png", "jpg", "jpeg"],
            )

            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                img = read_rgb_image(file_bytes, is_path=False)

                st.image(img, caption=uploaded_file.name, use_container_width=True)

                if st.button("Use This Image"):
                    st.session_state.selected_image = img
                    st.session_state.selected_image_name = uploaded_file.name
                    st.session_state.selected_image_source = "upload"
                    st.session_state.show_prediction = False
                    st.session_state.scroll_to_preview = True
                    st.rerun()

        # -------------------- Selected Image Preview + Predict Now --------------------
        if st.session_state.selected_image is not None:
            st.markdown("---")
            st.markdown('<div id="preview-anchor"></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-header">Selected Image Preview</p>', unsafe_allow_html=True)

            preview_img = cv2.resize(st.session_state.selected_image, (IMG_SIZE, IMG_SIZE))
            pc1, pc2, pc3 = st.columns([1, 1, 1])
            with pc2:
                st.image(preview_img, width=IMG_SIZE)
            st.write(f"**Filename:** `{st.session_state.selected_image_name}`")

            if st.button("🚀 Predict Now", use_container_width=True, type="primary"):
                st.session_state.show_prediction = True
                st.rerun()

            if st.session_state.scroll_to_preview:
                st.session_state.scroll_to_preview = False
                st.markdown(
                    """
                    <script>
                    setTimeout(function() {
                        var anchor = window.parent.document.getElementById('preview-anchor');
                        if (anchor) {
                            anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
                        } else {
                            window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
                        }
                    }, 200);
                    </script>
                    """,
                    unsafe_allow_html=True,
                )