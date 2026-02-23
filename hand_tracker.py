import cv2
import mediapipe as mp
import os

# New Tasks API imports
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand landmark connections for drawing (index pairs)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

class HandTracker:
    def __init__(self, max_hands=1):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.landmarks = []

    def find_hands(self, frame):
        """Detect hands, draw landmarks, and return the annotated frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        self.landmarks = []
        h, w, _ = frame.shape

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                # Convert normalized coords → pixel coords
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                self.landmarks = pts  # keep last detected hand

                # Draw connections
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)

                # Draw landmark dots
                for pt in pts:
                    cv2.circle(frame, pt, 5, (255, 0, 0), -1)

        return frame

    def get_landmarks(self):
        return self.landmarks

    def fingers_up(self):
        """Return a list of booleans [thumb, index, middle, ring, pinky]."""
        fingers = []
        if not self.landmarks:
            return []

        # Thumb: compare tip x vs joint x (for mirrored/flipped frame)
        fingers.append(self.landmarks[4][0] > self.landmarks[3][0])

        # Other four fingers: tip y < pip y means finger is up
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        for tip, pip in zip(tip_ids, pip_ids):
            fingers.append(self.landmarks[tip][1] < self.landmarks[pip][1])

        return fingers