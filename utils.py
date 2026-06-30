"""
utils.py

Helper functions for the AResUNet Streamlit app.

Preprocessing, inference, thresholding and metric calculations mirror the
original training notebook (completefinal.ipynb) exactly:

    - Image is converted to RGB
    - Resized to 512x512
    - Scaled to [0, 1] (float32 / 255.0)
    - Converted to a CHW tensor
    - Model output passed through sigmoid
    - Binarized using threshold = 0.5
"""

import time
import numpy as np
import cv2
import torch

# ---------------------------------------------------------------------
# Constants (taken directly from the notebook)
# ---------------------------------------------------------------------
IMG_SIZE = 512
THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Preprocessing — identical to BuildingDataset.__getitem__ in the notebook
# ---------------------------------------------------------------------
def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    """
    image_rgb: HxWx3 RGB uint8 numpy array
    Returns a 1x3xHxW float32 tensor scaled to [0, 1].
    """
    img = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def preprocess_mask(mask_gray: np.ndarray) -> np.ndarray:
    """
    mask_gray: HxW grayscale uint8 numpy array (ground truth mask)
    Returns a binarized (0/1) mask resized to IMG_SIZE x IMG_SIZE.
    """
    mask = cv2.resize(mask_gray, (IMG_SIZE, IMG_SIZE))
    mask = (mask > 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------
# Inference — identical logic to the notebook's evaluation cells
# ---------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, image_rgb: np.ndarray, device):
    """
    Runs the full inference pipeline on a single RGB image.

    Returns:
        binary_mask (HxW uint8, values 0/1, at IMG_SIZE resolution)
        inference_time_seconds (float)
    """
    model.eval()

    input_tensor = preprocess_image(image_rgb).to(device)

    start = time.time()
    logits = model(input_tensor)
    probs = torch.sigmoid(logits)
    pred = (probs > THRESHOLD).float()
    elapsed = time.time() - start

    binary_mask = pred.squeeze().cpu().numpy().astype(np.uint8)
    return binary_mask, elapsed


# ---------------------------------------------------------------------
# Metrics — identical formulas to the notebook
# ---------------------------------------------------------------------
def dice_score_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1).astype(np.float32)
    target = target.reshape(-1).astype(np.float32)
    inter = (pred * target).sum()
    return float((2 * inter + 1e-7) / (pred.sum() + target.sum() + 1e-7))


def iou_score_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1).astype(np.float32)
    target = target.reshape(-1).astype(np.float32)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter + 1e-7) / (union + 1e-7))


def accuracy_score_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    correct = (pred == target).sum()
    total = target.size
    return float(correct) / float(total)


def f1_score_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.reshape(-1).astype(np.float32)
    target = target.reshape(-1).astype(np.float32)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    f1 = (2 * precision * recall) / (precision + recall + 1e-7)
    return float(f1)


def building_percentage(mask: np.ndarray) -> float:
    """Percentage of pixels predicted as 'building' (value == 1)."""
    return float((mask == 1).sum()) / float(mask.size) * 100.0


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def make_overlay(image_rgb_resized: np.ndarray, binary_mask: np.ndarray,
                  color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    """
    image_rgb_resized: HxWx3 uint8 RGB image, same resolution as binary_mask
    binary_mask: HxW (0/1)
    Returns an HxWx3 uint8 RGB image with a transparent green overlay
    on predicted building pixels.
    """
    overlay = image_rgb_resized.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)
    color_layer[:, :] = color

    mask_bool = binary_mask.astype(bool)
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * color_layer[mask_bool]
    )

    return overlay.astype(np.uint8)


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    gt, pred: HxW (0/1) arrays of the same shape.

    Returns an HxWx3 uint8 RGB error map:
        Green  = True Positive
        Red    = False Positive
        Blue   = False Negative
        Black  = True Negative
    """
    comp = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

    comp[(gt == 1) & (pred == 1)] = [0, 255, 0]   # TP - Green
    comp[(gt == 0) & (pred == 1)] = [255, 0, 0]   # FP - Red
    comp[(gt == 1) & (pred == 0)] = [0, 0, 255]   # FN - Blue
    comp[(gt == 0) & (pred == 0)] = [0, 0, 0]     # TN - Black

    return comp


def mask_to_display(binary_mask: np.ndarray) -> np.ndarray:
    """Convert a 0/1 binary mask into a displayable 0/255 grayscale image."""
    return (binary_mask * 255).astype(np.uint8)
