# DeepFake Studio
Flask application with a premium dark interface, image inspection, video frame sampling and OpenCV face location.

Run locally:
pip install -r requirements.txt
python app.py

Deployment: Render + Gunicorn.

Important: Face detection is not deepfake classification. No real/fake prediction is made until a trained classifier is integrated and evaluated.
