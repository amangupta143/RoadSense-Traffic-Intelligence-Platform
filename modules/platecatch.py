"""
PlateCatch — License Plate Recognition Module
Part of RoadSense Traffic Intelligence Platform

Detects and extracts text from vehicle number plates using:
- OpenCV for image preprocessing and contour detection
- EasyOCR for text recognition
- imutils for contour utilities
"""

import cv2
import os
import numpy as np
import easyocr
import imutils


def run_plate_scan(input_path: str, output_path: str) -> dict:
    """
    Detect and read a vehicle number plate from an image.

    Args:
        input_path:  Path to the input image file.
        output_path: Path where the cropped plate image will be saved.

    Returns:
        dict with keys:
            - plate_text (str):   Extracted plate number, or empty string.
            - confidence (float): OCR confidence score (0-1).
            - found (bool):       Whether a plate region was detected.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot read image at {input_path}")

    # --- Preprocessing ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # --- Edge Detection ---
    edged = cv2.Canny(bfilter, 30, 200)

    # --- Contour Detection: find the rectangular number plate region ---
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        # Fallback: save the grayscale image and report no plate found
        cv2.imwrite(output_path, gray)
        return {'plate_text': '', 'confidence': 0.0, 'found': False}

    # --- Mask & Crop the plate region ---
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)

    (x_coords, y_coords) = np.where(mask == 255)
    x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
    x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    # --- OCR ---
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    ocr_results = reader.readtext(cropped_image)

    # Save the cropped plate image
    cv2.imwrite(output_path, cropped_image)

    if not ocr_results:
        return {'plate_text': '', 'confidence': 0.0, 'found': True}

    # Pick the highest-confidence result
    best = max(ocr_results, key=lambda r: r[2])
    plate_text = best[1].strip().upper()
    confidence = round(float(best[2]), 3)

    return {
        'plate_text': plate_text,
        'confidence': confidence,
        'found': True
    }
