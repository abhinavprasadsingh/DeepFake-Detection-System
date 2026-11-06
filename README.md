# DeepFake Detection System

## About the project

I am building this project to understand how deepfake detection
works and how machine learning can be used to examine images
and videos.

The idea is simple: a user uploads an image or video, and the
system will analyse it to check for signs of possible manipulation.

## What I am working on

- Uploading images and videos
- Extracting frames from videos
- Preparing face images for the model
- Training a deep learning model
- Showing the analysis results on a website

## Technologies

- Python
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Pandas

## Running the project locally

Install the dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

## Deployment

The project includes a render.yaml file for deployment on Render.

## Current status

The website and upload interface are being prepared first.
Model training and prediction integration are the next steps.

## Note

A model prediction is only an estimate. It cannot guarantee
whether an image or video is genuine or manipulated.
