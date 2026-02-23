from config import COLORS

class GestureController:
    def __init__(self):
        self.current_color = COLORS["BLUE"]
        self.mode = "DRAW"

    def detect(self, fingers):
        if not fingers:
            return "NONE"

        total = sum(fingers)

        # Clear canvas (all 5 fingers up) — check first
        if total == 5:
            return "CLEAR"

        # Erase mode (index + middle fingers up)
        if total == 2 and fingers[1] and fingers[2]:
            self.mode = "ERASE"
            return "ERASE"

        # Draw mode (index finger only)
        if total == 1 and fingers[1]:
            self.mode = "DRAW"
            return "DRAW"

        # Color selection (only one non-index finger up)
        if total == 1:
            if fingers[0]:
                self.current_color = COLORS["RED"]
            elif fingers[2]:
                self.current_color = COLORS["BLUE"]
            elif fingers[3]:
                self.current_color = COLORS["YELLOW"]
            elif fingers[4]:
                self.current_color = COLORS["PURPLE"]
            return "COLOR"

        return "NONE"