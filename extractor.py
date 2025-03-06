import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import g2o

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractPose(F):
    #Used for computing the rotation matrix
    W = np.mat([[0,-1,0],
                [1, 0, 0],
                [0, 0, 1]])
    
    #perform svd on the fundamental matrix
    U, d, Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0

    #correct vt if its determinant is negative
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    #compute initial rotation R using U, W and Vt
    R = np.dot(np.dot(U, W), Vt)

    #Check diagonal sum of R to ensure proper rotation matrix 
    #if not recompute R with transpose of W
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)

    # extract translation from 3rd column of U
    t = U[:, 2]

    #initialize a 4x4 identity matrix to store the pose
    ret = np.eye(4)

    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

def extract(img):
    orb = cv2.ORB_create()

    #Detection
    pts = cv2.goodFeaturesToTrack(image=np.mean(img, axis=-1).astype(np.uint8),
                                  maxCorners=1000,
                                  qualityLevel=0.01,
                                  minDistance=10)
    
    #Extraction
    kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
    kps, des = orb.compute(img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def match_frames(f1, f2):
    # k nearest neighbours matching on feature descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches contains pairs of matches (best match and second best match) for each feature
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    #applies Lowe's ratio test to filter out good matches based on 
    #distance threshold
    ret = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            # retrieve the actual point coordinates using the match indices
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            #additional distance test between p1 and p2
            if np.linalg.norm(p1-p2) < 0.1:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))
    
    #ensures that there are atleast 8 good matches (minimum required for fundamental matrix estim.)
    assert len(ret) >= 10
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Fit matrix
    model, inliers = ransac((ret[:,0], # matched points from the first image
                             ret[:,1]), #matched points from the second image
                             FundamentalMatrixTransform,
                             min_samples=10,
                             residual_threshold=0.005,
                             max_trials=200)
    
    # Ignore outliers
    ret = ret[inliers]
    Rt = extractPose(model.params)

    return idx1[inliers], idx2[inliers], Rt
    

def normalize(Kinv, pts):
    # The inverse camera intrinsic matrix 𝐾^(−1) transforms 2D homogeneous points 
    # from pixel coordinates to normalized image coordinates. 

    #This transformation centers the points based on the principal point(cx, cy) and scales
    #them according to focal lengths fx, fy, effectively mapping the points to a normalized
    #coordinate system where the principal point becomes the origin and the distances are 
    #scaled by the focal lengths
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    #converts normalized point to pixel coordinates by applying the intrinsic camera matrix and 
    #Normalizing the result
    ret = np.dot(K, [pt[0], pt[1], 1.0])
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

class Frame():
    def __init__(self, mapp, img, K):
        self.K = K #intrinsic camera matrix
        self.Kinv = np.linalg.inv(K) #Inverse of intrinsic camera matrix
        self.pose = np.eye(4) #Initial pose of the frame 

        self.id = len(mapp.frames) #add unique id for the frame based on number of frames in the map
        mapp.frames.append(self) #add this frame to the maps list of frames

        pts, self.des = extract(img) #extract feature points and descriptors from the image
        self.pts = normalize(self.Kinv, pts) #normalize the feature points using the inverse intrinsic matrix