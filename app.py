"""
RoadSense — Unified Traffic Intelligence Platform
Entry point for the Flask web application.
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from modules.platecatch import run_plate_scan
from modules.velocitycam import run_speed_vision
from modules.urbanpulse import run_street_scanner

app = Flask(__name__, static_folder="outputs", static_url_path="/outputs")
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_IMAGE = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, types):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in types


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/platecatch', methods=['POST'])
def api_platecatch():
    """PlateCatch — License Plate OCR endpoint. Accepts an image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    if not f or not allowed_file(f.filename, ALLOWED_IMAGE):
        return jsonify({'error': 'Invalid file. Provide an image (jpg, png, etc.)'}), 400

    uid = str(uuid.uuid4())[:8]
    filename = secure_filename(f.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{filename}")
    f.save(input_path)

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"platecatch_{uid}.jpg")

    try:
        result = run_plate_scan(input_path, output_path)
        return jsonify({
            'module': 'PlateCatch',
            'plate_text': result.get('plate_text', 'Not detected'),
            'confidence': result.get('confidence', 0),
            'output_image': f"/outputs/platecatch_{uid}.jpg",
            'uid': uid
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/velocitycam', methods=['POST'])
def api_velocitycam():
    """VelocityCam — Vehicle Speed Estimation endpoint. Accepts a video."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    if not f or not allowed_file(f.filename, ALLOWED_VIDEO):
        return jsonify({'error': 'Invalid file. Provide a video (mp4, avi, etc.)'}), 400

    uid = str(uuid.uuid4())[:8]
    filename = secure_filename(f.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{filename}")
    f.save(input_path)

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"velocitycam_{uid}.mp4")

    try:
        result = run_speed_vision(input_path, output_path)
        return jsonify({
            'module': 'VelocityCam',
            'vehicles_tracked': result.get('vehicles_tracked', 0),
            'avg_speed_kmh': result.get('avg_speed_kmh', 0),
            'max_speed_kmh': result.get('max_speed_kmh', 0),
            'speed_data': result.get('speed_data', []),
            'output_video': f"/outputs/velocitycam_{uid}.mp4",
            'uid': uid
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/urbanpulse', methods=['POST'])
def api_urbanpulse():
    """UrbanPulse — Semantic Segmentation endpoint. Accepts an image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    if not f or not allowed_file(f.filename, ALLOWED_IMAGE):
        return jsonify({'error': 'Invalid file. Provide an image (jpg, png, etc.)'}), 400

    uid = str(uuid.uuid4())[:8]
    filename = secure_filename(f.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{filename}")
    f.save(input_path)

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"urbanpulse_{uid}.png")

    try:
        result = run_street_scanner(input_path, output_path)
        return jsonify({
            'module': 'UrbanPulse',
            'pedestrian_count': result.get('pedestrian_count', 0),
            'vehicle_count': result.get('vehicle_count', 0),
            'pedestrian_pixel_pct': result.get('pedestrian_pixel_pct', 0),
            'vehicle_pixel_pct': result.get('vehicle_pixel_pct', 0),
            'output_image': f"/outputs/urbanpulse_{uid}.png",
            'uid': uid
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
