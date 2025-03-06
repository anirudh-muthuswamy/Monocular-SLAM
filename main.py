import cv2
import numpy as np
import gtsam
from pointmap import Map, Point
from display import Display
from extractor import Frame, match_frames, add_ones, denormalize
from multiprocessing import set_start_method, active_children

def pose3_to_matrix(pose3):
    R = pose3.rotation().matrix()
    t = pose3.translation()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def bundle_adjustment(pose1, pose2, pts4d, keypoints_2d, K):

    #homoegenous coordinates [x,y,z,w] are converted to euclidean coordinates
    pts3d_euc =  pts4d[:, :3] /pts4d[:, 3:]
    
    #convert intrinsic matrix to gtsam format
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1, 2]
    gtsam_K = gtsam.Cal3_S2(fx, fy, 0, cx, cy)

    #initialize factor graph and values
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    # Convert 4x4 transformation matrices to GTSAM Pose3
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]
    cam1_pose = gtsam.Pose3(gtsam.Rot3(R1), gtsam.Point3(t1))
    cam2_pose = gtsam.Pose3(gtsam.Rot3(R2), gtsam.Point3(t2))

    # Add camera poses to initial estimates
    initial_estimates.insert(0, cam1_pose)
    initial_estimates.insert(1, cam2_pose)

    # Fix first camera to remove gauge freedom
    graph.add(gtsam.PriorFactorPose3(0, cam1_pose, gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)))

    # Add 3D points to initial estimates
    for i, point in enumerate(pts3d_euc):
        initial_estimates.insert(i + 2, gtsam.Point3(point))

   # Add reprojection factors
    huber = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.0),
        gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    )
    for cam_idx, keypoints in enumerate(keypoints_2d):
        for i, (u, v) in enumerate(keypoints):
            point_key = i + 2
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(u, v), huber, cam_idx, point_key, gtsam_K))

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
    result = optimizer.optimize()

    # Extract results
    optimized_poses = [result.atPose3(i) for i in range(2)]
    optimized_points = np.array([result.atPoint3(2 + i) for i in range(len(pts4d))])

    return optimized_poses, optimized_points

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

def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return
    
    #previous frame f2 to the current frame f1
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)

    # X_f1 = E * X_f2, multiplying that with Rt transforms f2 pose wrt the f1 coordinate frame
    f1.pose = np.dot(Rt, f2.pose)

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])

    print("before optimization:\n")
    print("pts4d:",pts4d[0:5])
    print("euclidean:")
    temp = pts4d / pts4d[:, 3:]
    print(temp[0:5])
    print("pose1")
    print(f1.pose)
    print("pose2")
    print(f2.pose)

    # Run Bundle Adjustment
    optimized_poses, optimized_points = bundle_adjustment(f1.pose, f2.pose, pts4d, [f2.pts[idx2], f1.pts[idx1]], K)

    # Update camera poses and 3D points with optimized values
    # Convert optimized poses back to 4x4 transformation matrices
    f1.pose = pose3_to_matrix(optimized_poses[0])
    f2.pose = pose3_to_matrix(optimized_poses[1])

    print("After optimization:\n", optimized_points[0:5])

    pts4d = np.hstack((optimized_points, np.ones((optimized_points.shape[0], 1))))

    print(pts4d[0:5])
    print(f1.pose, f2.pose)

    # Reject points without enough "Parallax" and points behind the camera
    # returns, A boolean array indicating which points satisfy both criteria.
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
 
    for i, p in enumerate(pts4d):
        # If the point is not good (i.e., good_pts4d[i] is False), 
        # the loop skips the current iteration and moves to the next point.
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)
 
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
 
        cv2.circle(img, (u1,v1), 3, (0,255,0))
        cv2.line(img, (u1,v1), (u2, v2), (255,0,0))

    # #2d display
    # display.paint(img)

    # #3d display
    # mapp.display()


def kill_all_processes_and_exit():
    # Get all active child processes
    active_processes = active_children()
    
    # Terminate all active child processes
    for child in active_processes:
        child.terminate()
    
    # Wait for all child processes to close
    for child in active_processes:
        child.join()
    
    # Exit the program
    exit()

if __name__ == "__main__":
    set_start_method('spawn')
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
    mapp = Map()
    mapp.create_viewer()

    cap = cv2.VideoCapture("test_countryroad.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            kill_all_processes_and_exit()