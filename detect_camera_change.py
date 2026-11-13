import cv2
import numpy as np


class DetectCameraChange:
    def __init__(self):
        self.HIST_SIZE = 256
        self.BUFFER_SIZE = 60

        self.prev_hist = None
        self.similarity_buffer = np.zeros(self.BUFFER_SIZE)

    def detect_camera_change(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [self.HIST_SIZE], [0, 256])

        if self.prev_hist is None:
            self.prev_hist = hist
            return False

        score = cv2.compareHist(
            hist,
            self.prev_hist,
            cv2.HISTCMP_CORREL
        )

        self.prev_hist = hist

        return score < 0.5




