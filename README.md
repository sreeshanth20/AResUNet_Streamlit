# AResUNet — Building Segmentation (Streamlit App)

A production-ready Streamlit application for **AResUNet**, a custom Attention
Residual U-Net for building segmentation from satellite/aerial imagery.
The architecture, preprocessing, and inference pipeline are taken directly
from the original training notebook (`completefinal.ipynb`) — nothing was
modified or simplified.

## Project Structure

```
AResUNet_Streamlit/
│── app.py                 # Streamlit application
│── model.py                # AResUNet architecture (exact copy from notebook)
│── utils.py                 # Preprocessing, inference, metrics, overlay helpers
│── requirements.txt
│── README.md
│
│── model/                  # Place your checkpoint here
│── sample_images/           # Place sample images here
│── sample_masks/            # Place corresponding ground-truth masks here
│── outputs/                 # Saved outputs (created automatically)
│── assets/                  # Optional UI assets (logos, icons, etc.)
```

## 1. Installation

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Dependencies

- `streamlit` — web app framework
- `torch`, `torchvision` — model inference
- `segmentation-models-pytorch` — ResNet-34 encoder used inside AResUNet
- `opencv-python-headless` — image preprocessing
- `numpy`, `Pillow`, `pandas`

## 2. Add Your Model

Copy your trained checkpoint into the `model/` folder, named exactly:

```
model/newmodel333 (2).pth
```

If the file is missing, the app will show a clear error message on the
**Prediction** page (and elsewhere it's needed) instead of crashing.

> Note: `model.py` creates the ResNet-34 encoder with `encoder_weights=None`
> (no internet download of ImageNet weights is required), since your trained
> checkpoint already contains the full set of learned weights and will
> overwrite them on load.

## 3. Add Sample Images

Copy any `.png`, `.jpg`, `.jpeg`, or `.tif` images into:

```
sample_images/
```

They will automatically appear on the **Sample Images** page.

## 4. Add Sample Masks (optional, for metrics)

To automatically compute Dice Score, IoU, Pixel Accuracy, F1 Score, and
generate an error map for a sample image, copy a ground-truth mask into:

```
sample_masks/
```

The mask file must have **the same base filename** as its corresponding
image (extension can differ), e.g.:

```
sample_images/austin1.tif
sample_masks/austin1.png
```

## 5. Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually
`http://localhost:8501`).

## Features

- **Home** — overview of the app and model
- **About Model** — architecture details, training configuration, checkpoint info
- **Sample Images** — browse and select images bundled with the app
- **Upload Image** — upload your own PNG/JPG/JPEG image
- **Prediction** — run inference and view:
  - Original image, predicted mask, and green transparent overlay
  - Inference time, device used, image resolution, building percentage
  - Dice Score, IoU, Pixel Accuracy (when a matching ground-truth mask exists)
  - Color-coded error map (Green = TP, Red = FP, Blue = FN, Black = TN)
  - Download buttons for the prediction, binary mask, and overlay

## Notes

- The model is loaded once using Streamlit's `@st.cache_resource`, and
  automatically uses CUDA if available, falling back to CPU otherwise.
- All preprocessing (RGB conversion, resize to 512×512, normalization to
  [0, 1]) and the 0.5 prediction threshold match the original notebook
  exactly.
