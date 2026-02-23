import cv2
import numpy as np
from config import BRUSH_THICKNESS, ERASER_THICKNESS, SMOOTHING_FACTOR

class DrawingEngine:
    def __init__(self, frame_shape):
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.prev_point = None
        self.smooth_point = None

    def smooth(self, current_point):
        if self.smooth_point is None:
            self.smooth_point = current_point
            return current_point

        x = int(
            SMOOTHING_FACTOR * current_point[0] +
            (1 - SMOOTHING_FACTOR) * self.smooth_point[0]
        )

        y = int(
            SMOOTHING_FACTOR * current_point[1] +
            (1 - SMOOTHING_FACTOR) * self.smooth_point[1]
        )

        self.smooth_point = (x, y)
        return self.smooth_point

    def draw(self, point, color, mode="DRAW"):
        if point is None:
            self.prev_point = None
            return

        point = self.smooth(point)

        if self.prev_point is None:
            self.prev_point = point
            return

        if mode == "DRAW":
            cv2.line(
                self.canvas,
                self.prev_point,
                point,
                color,
                BRUSH_THICKNESS
            )

        elif mode == "ERASE":
            cv2.line(
                self.canvas,
                self.prev_point,
                point,
                (0, 0, 0),
                ERASER_THICKNESS
            )

        self.prev_point = point

    def clear(self):
        self.canvas[:] = 0

    def overlay(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, self.canvas)

        return frame