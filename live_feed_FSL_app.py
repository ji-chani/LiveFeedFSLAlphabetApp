"""
Streamlit live-feed FSL Alphabet Classifier
--------------------------------------------
Sidebar lets you pick the classifier and tune MediaPipe
hand-detection parameters. Live video comes in through streamlit-webrtc,
which is what gives a continuous feed (st.camera_input only
grabs single snapshots).
"""

import glob
import os
import string

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

from modules import LiveFeedFSLAlphabet

# ------ GLOBAL VARIABLES ----------
LANDMARKER_MODEL_PATH = './resources/hand_landmarker.task'
TRAINED_CLASSIFIER_DIR = './trained_models'
SCALER_PATH = './trained_models/scaler.sav'
LABELS = [c for c in string.ascii_uppercase if c not in ("J", "Z")] 
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
REFERENCE_IMAGE_PATH = "./resources/fsl_alphabet_guide.png"
# ----------------------------------

st.set_page_config(page_title="FSL Alphabet Classifier", layout='wide')
st.title("🤘Live-feed FSL (Static) Alphabet Classifier")
st.caption("by:  CBJetomo")
st.markdown(
    """
    ***:rainbow[Mabuhay!]*** 

    Welcome to this playground where you can try signing the **:red[Filipino sign language (FSL) alphabet]**.
    
    Recognized as the national sign language of the Philippines, FSL is different from other sign languages
    as it is tailor-fitted to the Filipino identity. 
    With its complex nature and structure, it offers a rich 
    look into varieties of culture of Philippine communities.

    This web tool is built as a proof-of-concept showing how machine learning
    can be used to recognize FSL. Recognized letters are limited to **:red[static FSL alphabet]** for now.


    *Users are encouraged to read our [paper](https://peerj.com/articles/cs-2720/)
    on this project. For any questions, kindly reach out through this [email](mailto:chanjetomo@gmail.com).*
    """
)
st.markdown("---")

# ------------- SIDEBAR --------------
st.sidebar.header("Settings")
st.sidebar.caption(
    "Change settings, then click **Apply and Restart Stream**"
    " to reload the pipeline new values."
)
st.sidebar.markdown("---")

classifier_files = sorted(glob.glob(os.path.join(TRAINED_CLASSIFIER_DIR, "*.sav")))
classifier_names = [
    os.path.basename(f) for f in classifier_files if os.path.basename(f) != os.path.basename(SCALER_PATH)
]
if not classifier_names:
    st.sidebar.error(f"No classifier .sav files found in {TRAINED_CLASSIFIER_DIR}")
    st.stop()


def reformat_classifier_file(classifier:str):
    if classifier.endswith('.sav'):
        return classifier[:-4]  # removes file extension
    else:
        return classifier+".sav"  # adds file extension
    

CLASSIFIER = st.sidebar.selectbox("Classifier", [reformat_classifier_file(clsf) for clsf in classifier_names])
CLASSIFIER = reformat_classifier_file(CLASSIFIER)
CLASSIFIER_PATH = os.path.join(TRAINED_CLASSIFIER_DIR, CLASSIFIER)

MAX_HANDS = st.sidebar.slider("Max hands", 1,2,2)  # min, max, default
DETECTION_CONF = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)  # min, max, default, step
TRACKING_CONF = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, 0.5, 0.05)  # min, max, default, step
SHOW_LANDMARKS = st.sidebar.checkbox("Show Landmarks", value=True)

APPLY_CLICKED = st.sidebar.button("Apply and Restart Stream")
# --------------------------------------

# The VideoProcessor is only re-instantiated when webrtc_streamer's `key`
# changes, so settings are captured via a closure and applied by bumping
# a counter in session_state, which forces a remount.
if "config_key" not in st.session_state:
    st.session_state.config_key = 0
if APPLY_CLICKED:
    st.session_state.config_key += 1


class FSLAlphabetVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.classifier = LiveFeedFSLAlphabet(
            model_path=LANDMARKER_MODEL_PATH,
            scaler_path=SCALER_PATH,
            classifier_path=CLASSIFIER_PATH,
            labels=LABELS,
            max_hands=MAX_HANDS,
            detection_conf=DETECTION_CONF,
            tracking_conf=TRACKING_CONF,
        )

        self.frame_ts_ms = 0

    def recv(self, frame):
        img = frame.to_ndarray(format='bgr24')
        img = cv2.flip(img,1)  # flip horizontally, matching process for LiveFeedFSLAlphabet

        annotated = self.classifier.classify_hand_to_letter(
            img, self.frame_ts_ms, show_landmarks=SHOW_LANDMARKS,
        )

        self.frame_ts_ms += 33  # -30fps timestamp increment

        return av.VideoFrame.from_ndarray(annotated, format='bgr24')

col1, col2 = st.columns(2)

with col1:
    st.caption("Live-feed Classifier")
    webrtc_streamer(
        key=f"fsl-alphabet-{st.session_state.config_key}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FSLAlphabetVideoProcessor,
        media_stream_constraints={
            "width": {"ideal": VIDEO_WIDTH},
            "height": {"ideal": VIDEO_HEIGHT}
        }
    )

with col2:
    if os.path.exists(REFERENCE_IMAGE_PATH):
        st.caption("FSL Static Alphabet Guide")
        st.image(REFERENCE_IMAGE_PATH, width=VIDEO_WIDTH)
    else:
        st.warning(f"Reference image not found at {REFERENCE_IMAGE_PATH}.")