import os
from flask import Flask, request, render_template, send_from_directory, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# Config
MODEL_PATH = "yolov8n.pt"   # put your model here
UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Load model (auto device)
model = YOLO(MODEL_PATH)  # ultralytics will use GPU if available

# Class names — keep in sync with your data.yaml / model training
CLASS_NAMES = [
    "bicycle","bus","car","cng","auto","bike","Multi-Class","rickshaw","truck","van"
]  

app = Flask(__name__)

from PIL import Image, ImageFont, ImageDraw
import numpy as np

def draw_boxes_on_image(image_path, boxes, scores, classes, save_path, conf_thres=0.0):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    # Ensure boxes is iterable (np.array -> list)
    if isinstance(boxes, np.ndarray):
        boxes_iter = boxes.tolist()
    else:
        boxes_iter = boxes

    for i, box in enumerate(boxes_iter):
        # safety for empty arrays or mismatched lengths
        conf = float(scores[i]) if i < len(scores) else 0.0
        cls = int(classes[i]) if i < len(classes) else 0

        if conf < conf_thres:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"

        # Draw bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Compute text size robustly
        try:
            # Pillow >= 8.0
            bbox = draw.textbbox((x1, y1), label, font=font)  # (x0,y0,x1,y1)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            try:
                # fallback
                text_w, text_h = font.getsize(label)
            except Exception:
                # final fallback estimate
                text_w, text_h = len(label) * 6, 12

        # Draw label background and text
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="red")
        draw.text((x1 + 2, y1 - text_h - 2), label, fill="white", font=font)

    img.save(save_path)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Save upload
    fname = file.filename
    upload_path = os.path.join(UPLOAD_DIR, fname)
    file.save(upload_path)

    # Run model
    # returns list of Results; passing device isn't necessary if model auto-detects GPU
    results = model.predict(source=upload_path, imgsz=640, conf=0.25, max_det=300, verbose=False)

    # Parse first result
    res = results[0]
    boxes_xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else np.empty((0,4))
    scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.empty((0,))
    classes = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else np.empty((0,))

    # Save annotated image
    out_fname = f"res_{fname}"
    out_path = os.path.join(RESULT_DIR, out_fname)
    draw_boxes_on_image(upload_path, boxes_xyxy, scores, classes, out_path, conf_thres=0.0)

    # Build JSON response
    detections = []
    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, scores, classes):
        detections.append({
            "class_id": int(cls),
            "class_name": CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(int(cls)),
            "confidence": float(conf),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    return jsonify({
        "uploaded": f"/{upload_path}",
        "result_image": f"/{out_path}",
        "detections": detections
    })

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    # Run locally
    app.run(host="0.0.0.0", port=5000, debug=True)
