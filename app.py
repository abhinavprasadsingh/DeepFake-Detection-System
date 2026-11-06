import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="DeepFake Detection",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background: #10111B;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
.hero {
    padding: 28px;
    border: 1px solid #34364A;
    border-radius: 18px;
    background: linear-gradient(120deg, #22233C, #171827);
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 36px;
    margin-bottom: 8px;
}
.hero p {
    color: #C4C5D5;
    font-size: 16px;
}
.info-card {
    background: #1B1D2B;
    padding: 20px;
    border: 1px solid #34364A;
    border-radius: 14px;
    min-height: 130px;
}
.small-text {
    color: #A8AABD;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("?? DeepFake AI")
    st.caption("Image and video analysis")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Home", "Image Analysis", "Video Analysis", "About"]
    )
    st.divider()
    st.caption("A student project built using Python and machine learning.")

if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>DeepFake Detection System</h1>
        <p>
        A simple tool to explore whether an image or video may
        contain AI-generated or manipulated faces.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome!")

    st.write(
        "Online images and videos can be edited in ways that are "
        "difficult to notice. This project aims to make it easier "
        "to examine media using machine learning."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>??? Image Analysis</h3>
            <p class="small-text">
            Upload an image and prepare it for deepfake analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>?? Video Analysis</h3>
            <p class="small-text">
            Upload a video and prepare its frames for analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info(
        "The detection model is still being developed. "
        "Predictions will be available after training and integration."
    )

elif page == "Image Analysis":
    st.title("Image Analysis")
    st.write("Upload an image containing a face to get started.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        st.info(
            "Your image has been uploaded successfully. "
            "The trained detection model will be connected here next."
        )
    else:
        st.caption("Supported formats: JPG, JPEG and PNG.")

elif page == "Video Analysis":
    st.title("Video Analysis")
    st.write("Upload a short video for frame-based analysis.")

    uploaded = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded:
        st.video(uploaded)
        st.info(
            "Video upload is working. Frame extraction and "
            "deepfake predictions will be added after model training."
        )
    else:
        st.caption("Supported formats: MP4, MOV and AVI.")

elif page == "About":
    st.title("About this project")

    st.write("""
    This project is being developed as part of my learning in
    artificial intelligence and machine learning.

    The main goal is to understand how deepfake detection works
    and build a practical application around it.
    """)

    st.subheader("Technologies used")

    st.write("""
    - Python
    - Machine Learning
    - PyTorch (for model training)
    - OpenCV (for image and video processing)
    - Streamlit (for the website)
    """)

    st.subheader("A small note")

    st.write(
        "The result of an AI detector should not be treated as "
        "absolute proof. Detection models can make mistakes, "
        "especially with unfamiliar or heavily edited media."
    )

st.divider()
st.caption("DeepFake Detection System | Student ML Project")
