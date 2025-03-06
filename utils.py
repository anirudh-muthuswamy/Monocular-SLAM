import numpy as np

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def pose3_to_matrix(pose3):
    R = pose3.rotation().matrix()
    t = pose3.translation()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def triangulate(pose1, pose2, pts1, pts2):
    #initialize result array to store homogenous coordinates
    ret = np.zeros((pts1.shape[0], 4))

    #invert the camera poses to ge the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)

    #loop through each pair of corresponding points
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        #initialize the matrix A to hold the linear equations
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        # The solution is the last row of V transposed (V^T), corresponding to the 
        # smallest singular value
        ret[i] = vt[3]

    return ret