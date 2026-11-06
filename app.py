import os, io, json, sqlite3, uuid, tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

BASE = Path(__file__).resolve().parent
DB = BASE / "instance" / "studio.db"
MODEL = "dima806/deepfake_vs_real_image_detection"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def database():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            filename TEXT,
            media_type TEXT,
            verdict TEXT,
            fake_score REAL,
            faces INTEGER,
            frames INTEGER,
            created TEXT,
            details TEXT
        )
    """)
    con.commit()
    return con

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    return faces

def model_prediction(image_bytes):
    token = os.environ.get("HF_TOKEN", "").strip()

    if not token:
        return {
            "available": False,
            "message": "HF_TOKEN is not configured. Face analysis is available, but AI classification is not enabled."
        }

    try:
        response = requests.post(
            HF_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream"
            },
            data=image_bytes,
            timeout=90
        )

        if response.status_code != 200:
            return {
                "available": False,
                "message": f"AI provider returned HTTP {response.status_code}. Check your Hugging Face token or provider availability."
            }

        output = response.json()

        if isinstance(output, dict):
            output = output.get("output", output.get("data", []))

        if not isinstance(output, list) or not output:
            return {"available": False, "message": "The AI provider returned an unexpected response."}

        predictions = []
        for item in output:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0))
            predictions.append({"label": label, "score": score})

        fake = max(
            [p["score"] for p in predictions if "fake" in p["label"]],
            default=None
        )
        real = max(
            [p["score"] for p in predictions if "real" in p["label"]],
            default=None
        )

        if fake is None and real is None:
            return {"available": False, "message": "Model labels could not be interpreted."}

        if fake is None:
            fake = 1.0 - real

        if real is None:
            real = 1.0 - fake

        verdict = "Potentially manipulated" if fake >= 0.5 else "Likely authentic"

        return {
            "available": True,
            "fake_score": round(fake * 100, 2),
            "real_score": round(real * 100, 2),
            "verdict": verdict,
            "model": MODEL,
            "warning": "Model predictions are estimates and are not proof of authenticity."
        }

    except requests.RequestException as exc:
        return {"available": False, "message": f"AI service connection failed: {str(exc)[:180]}"}
    except Exception as exc:
        return {"available": False, "message": f"Classification error: {str(exc)[:180]}"}

def analyze_frame(frame):
    faces = detect_faces(frame)
    height, width = frame.shape[:2]

    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = frame[y:y+h, x:x+w]
    else:
        crop = frame

    ok, encoded = cv2.imencode(
        ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )

    if not ok:
        raise ValueError("Could not prepare image for analysis.")

    prediction = model_prediction(encoded.tobytes())

    return {
        "face_count": int(len(faces)),
        "dimensions": f"{width} × {height}",
        "prediction": prediction
    }

def save_scan(filename, media_type, result):
    scan_id = str(uuid.uuid4())
    prediction = result.get("prediction", {})
    score = prediction.get("fake_score")

    if media_type == "video":
        available = [
            f["prediction"]["fake_score"]
            for f in result.get("frames", [])
            if f["prediction"].get("available")
        ]
        if available:
            score = round(sum(available) / len(available), 2)

    verdict = (
        "AI classification unavailable" if score is None
        else "Potentially manipulated" if score >= 50
        else "Likely authentic"
    )

    con = database()
    con.execute("""
        INSERT INTO scans
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_id, filename, media_type, verdict, score,
        result.get("faces", result.get("face_count", 0)),
        result.get("frames_analyzed", 0),
        datetime.now(timezone.utc).isoformat(),
        json.dumps(result)
    ))
    con.commit()
    con.close()
    return scan_id

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "DeepFake Studio",
        "classification_configured": bool(os.environ.get("HF_TOKEN")),
        "face_detection": not CASCADE.empty()
    })

@app.route("/api/status")
def status():
    con = database()
    count = con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    con.close()
    return jsonify({
        "status": "operational",
        "total_scans": count,
        "model": MODEL,
        "classification_configured": bool(os.environ.get("HF_TOKEN")),
        "face_detection": not CASCADE.empty()
    })

@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify({"error": "Please select an image."}), 400

    if Path(upload.filename).suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        return jsonify({"error": "Supported image formats: JPG, PNG, WEBP."}), 400

    raw = upload.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "The uploaded image could not be read."}), 400

    try:
        result = analyze_frame(image)
        result["filename"] = secure_filename(upload.filename)
        result["media_type"] = "image"
        result["faces"] = result["face_count"]
        scan_id = save_scan(result["filename"], "image", result)
        result["scan_id"] = scan_id
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify({"error": "Please select a video."}), 400

    if Path(upload.filename).suffix.lower() not in [".mp4", ".mov", ".avi", ".mkv"]:
        return jsonify({"error": "Supported formats: MP4, MOV, AVI, MKV."}), 400

    suffix = Path(upload.filename).suffix.lower()

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(upload.read())
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return jsonify({"error": "Could not open this video."}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if fps else 0

        if duration > 300:
            cap.release()
            return jsonify({"error": "Please upload a video shorter than 5 minutes."}), 400

        interval = max(1, int(fps * 2))
        frames = []
        total_faces = 0
        index = 0
        analyzed = 0

        while analyzed < 12:
            ok, frame = cap.read()
            if not ok:
                break

            if index % interval == 0:
                item = analyze_frame(frame)
                item["timestamp_seconds"] = round(index / fps, 2)
                total_faces += item["face_count"]
                frames.append(item)
                analyzed += 1

            index += 1

        cap.release()

        if not frames:
            return jsonify({"error": "No readable video frames were found."}), 400

        result = {
            "filename": secure_filename(upload.filename),
            "media_type": "video",
            "duration_seconds": round(duration, 2),
            "frames_analyzed": len(frames),
            "faces": total_faces,
            "frames": frames
        }

        scan_id = save_scan(result["filename"], "video", result)
        result["scan_id"] = scan_id
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/api/history")
def history():
    con = database()
    rows = con.execute(
        "SELECT id, filename, media_type, verdict, fake_score, faces, frames, created FROM scans ORDER BY created DESC LIMIT 50"
    ).fetchall()
    con.close()
    return jsonify([dict(row) for row in rows])

@app.route("/api/report/<scan_id>")
def report(scan_id):
    con = database()
    row = con.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    con.close()

    if not row:
        return jsonify({"error": "Report not found."}), 404

    data = dict(row)
    data["details"] = json.loads(data["details"])
    return send_file(
        io.BytesIO(json.dumps(data, indent=2).encode()),
        mimetype="application/json",
        as_attachment=True,
        download_name=f"deepfake-report-{scan_id[:8]}.json"
    )

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large. Maximum upload size is 25 MB."}), 413

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
