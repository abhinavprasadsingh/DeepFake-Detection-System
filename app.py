import os, io, tempfile
from base64 import b64encode
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def inspect_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(32, 32))
    preview = frame.copy()
    for x, y, w, h in faces:
        cv2.rectangle(preview, (x,y), (x+w,y+h), (90,225,175), 2)
    image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return len(faces), "data:image/jpeg;base64," + b64encode(buffer.getvalue()).decode()

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify(status="ok", service="DeepFake Studio")

@app.post("/api/analyze/image")
def analyze_image():
    f = request.files.get("file")
    if not f:
        return jsonify(error="Choose an image first."), 400
    if f.filename.rsplit(".",1)[-1].lower() not in ("png","jpg","jpeg","webp"):
        return jsonify(error="Unsupported image format."), 400
    try:
        im = Image.open(f.stream).convert("RGB")
        frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        count, preview = inspect_frame(frame)
        return jsonify(filename=f.filename, width=im.width, height=im.height,
                       faces=count, preview=preview,
                       notice="Face location only, not deepfake classification.")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="Could not read this image."), 400

@app.post("/api/analyze/video")
def analyze_video():
    f = request.files.get("file")
    if not f:
        return jsonify(error="Choose a video first."), 400
    ext = f.filename.rsplit(".",1)[-1].lower()
    if ext not in ("mp4","mov","avi","mkv"):
        return jsonify(error="Unsupported video format."), 400
    path = None
    cap = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+ext) as tmp:
            f.save(tmp)
            path = tmp.name
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return jsonify(error="Cannot open video."), 400
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return jsonify(error="No readable frames."), 400
        indices = np.linspace(0,total-1,min(8,total),dtype=int)
        samples = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
            ok, frame = cap.read()
            if ok:
                count, preview = inspect_frame(frame)
                samples.append(dict(frame=int(i),faces=count,preview=preview))
        return jsonify(total_frames=total,sampled_frames=len(samples),
                       frames_with_faces=sum(x["faces"]>0 for x in samples),
                       face_detections=sum(x["faces"] for x in samples),
                       samples=samples,
                       notice="Sampled-frame face location only, not deepfake classification.")
    finally:
        if cap: cap.release()
        if path and os.path.exists(path): os.remove(path)

@app.errorhandler(413)
def too_large(e):
    return jsonify(error="Maximum upload size is 40 MB."), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
