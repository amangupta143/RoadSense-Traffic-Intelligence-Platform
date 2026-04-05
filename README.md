# RoadSense — Traffic Intelligence Platform

A unified web application combining three computer vision modules for traffic analysis.

---

## Project Structure

```
RoadSense/
├── app.py                     # Flask web server & API routes
├── requirements.txt
├── templates/
│   └── index.html             # Single-page web UI
├── modules/
│   ├── platecatch.py          # License plate OCR (was: PlateScan)
│   ├── velocitycam.py         # Speed estimation (was: SpeedVision)
│   └── urbanpulse.py          # Scene segmentation (was: StreetScanner)
├── uploads/                   # Auto-created on first run
└── outputs/                   # Auto-created on first run
```

---

## Module Mapping

| New Name      | Input  | Function                          |
|---------------|--------|-----------------------------------|
| PlateCatch    | Image  | Number plate detection + OCR      |
| VelocityCam   | Video  | Vehicle tracking + speed in km/h  |
| UrbanPulse    | Image  | Semantic segmentation overlay     |

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## API Endpoints

### POST `/api/platecatch`
- Body: `multipart/form-data` with `file` (image)
- Returns: `plate_text`, `confidence`, `output_image` URL

### POST `/api/velocitycam`
- Body: `multipart/form-data` with `file` (video)
- Returns: `vehicles_tracked`, `avg_speed_kmh`, `max_speed_kmh`, `speed_data[]`, `output_video` URL

### POST `/api/urbanpulse`
- Body: `multipart/form-data` with `file` (image)
- Returns: `pedestrian_count`, `vehicle_count`, `pedestrian_pixel_pct`, `vehicle_pixel_pct`, `output_image` URL

---

## Notes

- **VelocityCam** requires `yolov8x.pt` — download automatically via `ultralytics` on first run, or place the weights file in the project root.
- **UrbanPulse** downloads DeepLabv3 weights from PyTorch Hub on first run (~100 MB).
- The `SOURCE` polygon in `velocitycam.py` is pre-set for a typical highway perspective shot. Adjust the coordinates to match your camera setup for accurate speed readings.
- GPU acceleration is optional — all three modules run on CPU by default.
