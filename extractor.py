import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import random

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def project_points(X_3d, R_mat, t, K):

    N = X_3d.shape[0]
    X_h = np.hstack((X_3d, np.ones((N, 1))))  # Convert to homogeneous
    P = K @ np.hstack((R_mat, t))  # Projection matrix

    X_proj = (P @ X_h.T).T  # Project points
    X_proj /= X_proj[:, 2:]  # Normalize homogeneous coordinates

    return X_proj[:, :2]  # Return 2D projection

def reprojection_error(params, X_3d, x_2d, K):

    q = params[:4]  # Extract quaternion
    t = params[4:].reshape(3, 1)  # Extract translation

    R_mat = R.from_quat(q).as_matrix()  # Convert quaternion to rotation matrix
    x_proj = project_points(X_3d, R_mat, t, K)

    return (x_proj - x_2d).ravel()  # Flatten error array


def nonlinear_pnp(X_3d, x_2d, K, R_init, t_init):

    # Convert rotation matrix to quaternion for optimization
    q_init = R.from_matrix(R_init).as_quat()
    t_init = t_init.flatten()  # Ensure translation is a flat array

    # Initial parameter vector [q0, q1, q2, q3, t1, t2, t3]
    initial_params = np.hstack((q_init, t_init))

    # Optimize using least squares
    result = least_squares(reprojection_error, initial_params, args=(X_3d, x_2d, K), method='lm')

    # Extract optimized values
    refined_q = result.x[:4]
    refined_t = result.x[4:].reshape(3, 1)

    refined_R = R.from_quat(refined_q).as_matrix()  # Convert quaternion back to rotation matrix

    return refined_R, refined_t

def extractPoseFromFundamentalMatrix(F):
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

def extractPosefromEssentialMatrix(E):
    # Perform SVD on Essential Matrix
    U, D, Vt = np.linalg.svd(E)

    # Enforce the rank-2 constraint (two singular values should be equal, third should be zero)
    D = np.diag([1, 1, 0])  # Essential matrix has two singular values set to 1, last to 0
    E = U @ D @ Vt  # Reconstruct the essential matrix

    # Ensure proper determinant sign of U and Vt
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Define W matrix for extracting rotation
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    # Compute the two possible rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Extract the translation vector
    t = U[:, 2]  # The third column of U gives the translation (up to scale)

    # Ensure R1 and R2 are valid rotation matrices (det(R) should be 1)
    if np.linalg.det(R1) < 0:
        R1 *= -1
    if np.linalg.det(R2) < 0:
        R2 *= -1

    return R1, R2, t

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


def matchStereoFrames(f, calibrated = True):
    # k nearest neighbours matching on feature descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches contains pairs of matches (best match and second best match) for each feature
    matches = bf.knnMatch(f.desLeft, f.desRight, k=2)

    #applies Lowe's ratio test to filter out good matches based on 
    #distance threshold
    ret = []
    idxLeft, idxRight = [], []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            # retrieve the actual point coordinates using the match indices
            pLeft = f.ptsLeft[m.queryIdx]
            pRight = f.ptsRight[m.trainIdx]
            #additional distance test between p1 and p2
            if calibrated:
                # In rectified images, corresponding points should have similar y-coordinates
                if abs(pLeft[1] - pRight[1]) < 1.0:  # 1-pixel threshold for y-coordinate
                    idxLeft.append(m.queryIdx)
                    idxRight.append(m.trainIdx)
                    ret.append((pLeft, pRight))
            else:
                print("TODO: PERFROM CALIBRATION FIRST")
                exit()
        
    #ensures that there are atleast 8 good matches (minimum required for fundamental matrix estim.)
    assert len(ret) >= 8
    ret = np.array(ret)
    idxLeft = np.array(idxLeft)
    idxRight = np.array(idxRight)

    # Fit matrix
    model, inliers = ransac((ret[:, 0], 
                                ret[:, 1]),
                               EssentialMatrixTransform,
                               min_samples=8,
                               residual_threshold=0.005,
                               max_trials=200)
    E = model.params
    
    # Ignore outliers
    ret = ret[inliers]
    R1, R2, t = extractPosefromEssentialMatrix(E)

    return idxLeft[inliers], idxRight[inliers], R1, R2, t

def match_frames(f1, f2):
    # k nearest neighbours matching on feature descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches contains pairs of matches (best match and second best match) for each feature
    matches = bf.knnMatch(f1.desLeft, f2.desLeft, k=2)

    #applies Lowe's ratio test to filter out good matches based on 
    #distance threshold
    ret = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            # retrieve the actual point coordinates using the match indices
            p1 = f1.ptsLeft[m.queryIdx]
            p2 = f2.ptsLeft[m.trainIdx]
            #additional distance test between p1 and p2
            if abs(p1[1] - p1[1]) < 1.0: 
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))
    
    #ensures that there are atleast 8 good matches (minimum required for fundamental matrix estim.)
    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Fit matrix
    _, inliers = ransac((ret[:,0], # matched points from the first image
                             ret[:,1]), #matched points from the second image
                             FundamentalMatrixTransform,
                             min_samples=8,
                             residual_threshold=0.005,
                             max_trials=200)

    return idx1[inliers], idx2[inliers]

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

class StereoFrame():
    def __init__(self, mapp, imgLeft, imgRight, KLeft, KRight):
        self.KLeft = KLeft #intrinsic camera matrix
        self.KRight = KRight
        self.KLeftinv = np.linalg.inv(KLeft) #Inverse of intrinsic camera matrix
        self.KRightinv = np.linalg.inv(KRight) #Inverse of intrinsic camera matrix
        self.point_indices = {}
        self.filtered_2dpt_idx = []
        self.pts_3d = None

        #pose is based on the left camera
        self.poseLeft = np.eye(4) #Initial pose of the frame 
        self.poseRight = np.eye(4)  # 4x4 matrix for the right camera


        self.idLeft = len(mapp.frames) #add unique id for the frame based on number of frames in the map
        mapp.frames.append(self) #add this frame to the maps list of frames

        ptsLeft, self.desLeft = extract(imgLeft) #extract feature points and descriptors from the image
        ptsRight, self.desRight = extract(imgRight)
        self.ptsLeft = normalize(self.KLeftinv, ptsLeft) #normalize the feature points using the inverse intrinsic matrix
        self.ptsRight = normalize(self.KRightinv, ptsRight)

    def setInitialRightPose(self, baseline):
        self.poseRight[0:3, 3] = np.array([baseline, 0, 0])

