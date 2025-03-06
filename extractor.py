import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import g2o

IRt = np.eye(4)

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
    print(len(pts))
    
    #Extraction
    kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
    kps, des = orb.compute(img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def normalize(Kinv, pts):
    # The inverse camera intrinsic matrix 𝐾^(−1) transforms 2D homogeneous points 
    # from pixel coordinates to normalized image coordinates. 

    #This transformation centers the points based on the principal point(cx, cy) and scales
    #them according to focal lengths fx, fy, effectively mapping the points to a normalized
    #coordinate system where the principal point becomes the origin and the distances are 
    #scaled by the focal lengths
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

class Frame():
    def __init__(self, mapp, img, K):
        self.K = K #intrinsic camera matrix
        self.Kinv = np.linalg.inv(K) #Inverse of intrinsic camera matrix
        self.pose = IRt #Initial pose of the frame (eye(4))

        self.id = len(mapp.frames) #add unique id for the frame based on number of frames in the map
        mapp.frames.append(self) #add this frame to the maps list of frames

        pts, self.des = extract(img) #extract feature points and descriptors from the image
        self.pts = normalize(self.Kinv, pts) #normalize the feature points using the inverse intrinsic matrix