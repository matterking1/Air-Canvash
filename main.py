import cv2
from hand_tracker import HandTracker
from gesture_controller import GestureController
from drawing_engine import DrawingEngine

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend for Windows

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check that your camera is connected.")
        return

    # Read a warm-up frame to get stable dimensions
    for _ in range(5):
        ret, frame = cap.read()

    if not ret or frame is None:
        print("[ERROR] Could not read from webcam.")
        cap.release()
        return

    frame = cv2.flip(frame, 1)

    tracker = HandTracker()
    gesture = GestureController()
    drawer = DrawingEngine(frame.shape)

    print("[INFO] Air Canvas started! Press 'q' to quit.")
    print("[INFO] Gestures:")
    print("       1 finger up  → select color  (Thumb=Red, Index=Green, Middle=Blue, Ring=Yellow, Pinky=Purple)")
    print("       2 fingers up → ERASE mode")
    print("       5 fingers up → CLEAR canvas")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARNING] Failed to grab frame, skipping...")
            continue

        frame = cv2.flip(frame, 1)

        frame = tracker.find_hands(frame)
        landmarks = tracker.get_landmarks()
        fingers = tracker.fingers_up()

        action = gesture.detect(fingers)

        if action == "CLEAR":
            drawer.clear()

        if landmarks and action in ("DRAW", "ERASE"):
            index_tip = landmarks[8]
            drawer.draw(index_tip, gesture.current_color, gesture.mode)
        else:
            drawer.draw(None, gesture.current_color, gesture.mode)

        frame = drawer.overlay(frame)

        # HUD overlay
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"Mode: {gesture.mode}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            gesture.current_color,
            2
        )

        color_names = ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE"]
        colors_bgr = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)]
        for i, (name, clr) in enumerate(zip(color_names, colors_bgr)):
            x = frame.shape[1] - 160 + i * 30
            cv2.circle(frame, (x, 30), 12, clr, -1)
            if clr == gesture.current_color:
                cv2.circle(frame, (x, 30), 14, (255, 255, 255), 2)

        cv2.imshow("AirCanvas - Shivam Edition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Air Canvas closed.")

if __name__ == "__main__":
    main()