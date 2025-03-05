import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import g2o

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractPose(F):
    pass

def extract(img):
    orb = cv2.ORB_create()

    #Detection
    pts = cv2.goodFeaturesToTrack(image=np.mean(img, axis=-1).astype(np.uint8),
                                  maxCorners=1000,
                                  qualityLevel=0.01,
                                  minDistance=10)
    print(pts)
    
    #Extraction
    kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]