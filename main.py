import cv2
import numpy as np
from pointmap import Map, Point
from display import Display
from extractor import Frame

#define the principle point offset or the optical center coordinates
W, H = 1920//2, 1080//2

#define the focal length
F=450

#define the intrinsic matrix and the inverse of the intrinsic matrix
K = np.array([[F, 0, W//2],
              [0, F, H//2],
              [0, 0, 1]])
Kinv = np.linalg.inv(K)

display = Display(W, H)

print('display created')

#initialize a map
mapp = Map()
mapp.create_viewer()


def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return
    print(img.shape)
    

if __name__ == "__main__":
    cap = cv2.VideoCapture("car.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break