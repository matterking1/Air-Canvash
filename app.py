import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import numpy as np

from hand_tracker import HandTracker
from gesture_controller import GestureController
from drawing_engine import DrawingEngine

# Setup page layout
st.set_page_config(page_title="Air Canvas", layout="wide", page_icon="🖌️")

st.title("🖌️ Air Canvas - AI Drawing App")
st.markdown("Draw on your screen using hand gestures right in your browser!")

with st.expander("📖 How to use (Gestures)"):
    st.markdown("""
    - ☝️ **1 finger up**: Draw mode
    - ✌️ **2 fingers up**: Erase mode
    - 🖐 **5 fingers up**: Clear canvas
    - 🖖 **Color Selection**:
        - Thumb = Red
        - Middle = Blue
        - Ring = Yellow
        - Pinky = Purple
    """)

# Streamlit-WebRTC config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class AirCanvasProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = None
        self.gesture = None
        self.drawer = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Initialize instances once we know the frame shape
        if self.tracker is None:
            self.tracker = HandTracker()
            self.gesture = GestureController()
            self.drawer = DrawingEngine(img.shape)

        # Flip horizontally for better mirror effect
        img = cv2.flip(img, 1)

        # Process hands
        img = self.tracker.find_hands(img)
        landmarks = self.tracker.get_landmarks()
        fingers = self.tracker.fingers_up()

        # Get action from gesture controller
        action = self.gesture.detect(fingers)

        if action == "CLEAR":
            self.drawer.clear()

        if landmarks and action in ("DRAW", "ERASE"):
            index_tip = landmarks[8]
            self.drawer.draw(index_tip, self.gesture.current_color, self.gesture.mode)
        else:
            self.drawer.draw(None, self.gesture.current_color, self.gesture.mode)

        # Overlay canvas drawing
        img = self.drawer.overlay(img)

        # HUD overlay
        cv2.rectangle(img, (0, 0), (img.shape[1], 60), (30, 30, 30), -1)
        cv2.putText(
            img,
            f"Mode: {self.gesture.mode}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            self.gesture.current_color,
            2
        )

        color_names = ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE"]
        colors_bgr = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)]
        for i, (name, clr) in enumerate(zip(color_names, colors_bgr)):
            x = img.shape[1] - 160 + i * 30
            cv2.circle(img, (x, 30), 12, clr, -1)
            # Highlight current color
            if clr == self.gesture.current_color:
                cv2.circle(img, (x, 30), 14, (255, 255, 255), 2)

        # Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start Streamlit-webrtc component
webrtc_streamer(
    key="air-canvas",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=AirCanvasProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
