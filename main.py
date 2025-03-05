import cv2
import numpy as np


def process_frame(img):
    pass

if __name__ == "__main__":
    cap = cv2.VideoCapture("path_to_video.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break