"""
UrbanPulse — Scene Segmentation Module
Part of RoadSense Traffic Intelligence Platform

Performs semantic segmentation to detect and colorize:
- Pedestrians (COCO class 15)
- Vehicles: car=7, bus=6, motorcycle=4, bicycle=2, truck=8

Uses DeepLabv3 with ResNet-50 backbone (pretrained on COCO).
"""

import os
import numpy as np
from PIL import Image

try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# COCO class indices relevant to traffic scenes
PEDESTRIAN_CLASS = 15     # person
VEHICLE_CLASSES = {2, 4, 6, 7, 8}   # bicycle, motorcycle, bus, car, truck

# Color palette — each of the 21 COCO classes gets a distinct color
COCO_PALETTE = [
    (0, 0, 0),        # 0  background
    (128, 0, 0),      # 1  aeroplane
    (0, 128, 0),      # 2  bicycle       — vehicle (green tint)
    (128, 128, 0),    # 3  bird
    (0, 0, 128),      # 4  boat
    (128, 0, 128),    # 5  bottle
    (0, 128, 128),    # 6  bus           — vehicle (teal)
    (128, 128, 128),  # 7  car           — vehicle (gray)
    (64, 0, 0),       # 8  cat
    (192, 0, 0),      # 9  chair
    (64, 128, 0),     # 10 cow
    (192, 128, 0),    # 11 dining table
    (64, 0, 128),     # 12 dog
    (192, 0, 128),    # 13 horse
    (64, 128, 128),   # 14 motorbike
    (57, 181, 255),   # 15 person        — pedestrian (bright blue)
    (0, 64, 0),       # 16 potted plant
    (128, 64, 0),     # 17 sheep
    (0, 192, 0),      # 18 sofa
    (128, 192, 0),    # 19 train
    (0, 64, 128),     # 20 tv/monitor
]

# Override vehicle classes with distinctive warm color, pedestrians with cyan
VEHICLE_COLOR = (255, 140, 0)    # orange
PEDESTRIAN_COLOR = (0, 220, 220) # cyan

for idx in VEHICLE_CLASSES:
    if idx < len(COCO_PALETTE):
        COCO_PALETTE[idx] = VEHICLE_COLOR
COCO_PALETTE[PEDESTRIAN_CLASS] = PEDESTRIAN_COLOR


def _build_flat_palette(palette):
    """Flatten a list of (R,G,B) tuples into a 768-element list for PIL putpalette."""
    flat = []
    for rgb in palette:
        flat.extend(rgb)
    # Pad to 256 colors
    while len(flat) < 768:
        flat.extend([0, 0, 0])
    return flat


def run_street_scanner(input_path: str, output_path: str) -> dict:
    """
    Run DeepLabv3 segmentation on a traffic scene image.

    Args:
        input_path:  Path to the input image (JPEG/PNG).
        output_path: Path where the color-coded segmentation mask is saved.

    Returns:
        dict with keys:
            - pedestrian_count (int):       Estimated number of pedestrian blobs.
            - vehicle_count (int):          Estimated number of vehicle blobs.
            - pedestrian_pixel_pct (float): % of image pixels labeled as person.
            - vehicle_pixel_pct (float):    % of image pixels labeled as vehicle.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch and torchvision are required for UrbanPulse. "
            "Install via: pip install torch torchvision"
        )

    # --- Load model ---
    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'deeplabv3_resnet50',
        pretrained=True
    )
    model.eval()

    # --- Preprocessing pipeline ---
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_image = Image.open(input_path).convert('RGB')
    input_tensor = preprocess(input_image).unsqueeze(0)

    # --- Inference ---
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    predictions = output.argmax(0).byte().cpu().numpy()
    total_pixels = predictions.size

    # --- Pixel-level class statistics ---
    pedestrian_mask = (predictions == PEDESTRIAN_CLASS)
    vehicle_mask = np.isin(predictions, list(VEHICLE_CLASSES))

    ped_pixels = int(pedestrian_mask.sum())
    veh_pixels = int(vehicle_mask.sum())

    ped_pct = round(ped_pixels / total_pixels * 100, 2)
    veh_pct = round(veh_pixels / total_pixels * 100, 2)

    # --- Estimate blob counts using connected components ---
    try:
        import cv2
        ped_blobs, _ = cv2.connectedComponents(pedestrian_mask.astype(np.uint8))
        veh_blobs, _ = cv2.connectedComponents(vehicle_mask.astype(np.uint8))
        # Subtract 1 for background component
        ped_count = max(0, ped_blobs - 1)
        veh_count = max(0, veh_blobs - 1)
    except ImportError:
        ped_count = 1 if ped_pixels > 0 else 0
        veh_count = 1 if veh_pixels > 0 else 0

    # --- Build color-coded output mask ---
    seg_image = Image.fromarray(predictions)
    seg_image.putpalette(_build_flat_palette(COCO_PALETTE))
    seg_image.save(output_path)

    # Also save a side-by-side composite (original + overlay)
    _save_composite(input_image, predictions, output_path)

    return {
        'pedestrian_count': ped_count,
        'vehicle_count': veh_count,
        'pedestrian_pixel_pct': ped_pct,
        'vehicle_pixel_pct': veh_pct,
    }


def _save_composite(original: Image.Image, predictions: np.ndarray, output_path: str):
    """
    Saves a blended overlay of the segmentation mask on the original image.
    Replaces the plain mask output_path with the composite.
    """
    # Build RGB mask
    h, w = predictions.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(COCO_PALETTE[:21]):
        mask_rgb[predictions == class_id] = color

    orig_resized = original.resize((w, h)).convert('RGB')
    orig_np = np.array(orig_resized)

    # Blend: 60% original, 40% mask
    composite = (orig_np * 0.6 + mask_rgb * 0.4).astype(np.uint8)
    Image.fromarray(composite).save(output_path)
