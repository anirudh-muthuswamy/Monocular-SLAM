import cv2
import numpy as np
from pointmap import Map, Point
from display import Display
from extractor import (add_ones, denormalize, matchStereoFrames, StereoFrame,
                       nonlinear_pnp, match_frames)
from multiprocessing import set_start_method, active_children
from non_linear_triangulation import refine_triangulation, filter_outliers
from utils import extract_intrinsic_matrix, read_calibration_file


def check_points_in_front(points_3d, R, t):
    # First camera is at origin with identity rotation
    count = 0

    for X in points_3d:
        # Check if point is in front of the first camera
        # For first camera, the point must have positive Z coordinate
        if X[2] <= 0:
            continue
            
        # Check if point is in front of the second camera
        # Transform point to second camera's coordinate system
        X_cam2 = R.dot(X[:3]) + t.reshape(3)
        
        # For second camera, the transformed point must have positive Z coordinate
        if X_cam2[2] > 0:
            count += 1
            
    return count

def triangulate_points_stereo(R0, t0, R_set, t_set, pts1, pts2, first_frame=True):

    if first_frame:
        # First camera matrix P1 = [I|0]
        P1 = np.hstack((R0, t0))
    
    best_solution = None
    max_points_in_front = 0
    
    # Test all four possible solutions
    for i in range(len(R_set)):
        R = R_set[i]
        t = t_set[i]

        print(R)
        print(t)
        # print(R)
        # print(t)
        
        # Second camera matrix P2 = [R|t]
        P2 = np.eye(4)  # Create 4x4 identity matrix
        P2[:3, :3] = R  # Set upper-left 3x3 block to rotation matrix
        if first_frame == False:
            P2[:3, 3] = t.T
        else:
            P2[:3, 3] = t   # Set upper-right 3x1 block to translation vector
        
        # Triangulate points
        points_3d = triangulate(K1, R, t, pts1, pts2)
        points_3d /= points_3d[:, 3:]
        
        # Check points in front of both cameras
        points_in_front = check_points_in_front(points_3d, R, t)
        
        if points_in_front > max_points_in_front:
            max_points_in_front = points_in_front
            best_solution = (R, t, points_3d[:,:3])
    
    return best_solution

def triangulate(K, R, t, pts1, pts2):
    #initialize result array to store homogenous coordinates
    ret = np.zeros((pts1.shape[0], 4))

    #get the projection matrices
    Rt = np.hstack((R, t.reshape(3, 1)))  # Concatenate R and t to form [R | t]
    pose1 = K @ Rt  # Matrix multiplication
    pose2 = K @ Rt

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

def add_to_map(imgLeft, currentFrame, pts_3d_euc, idx):
    #pts_3d_euc ids are from i 
    #idx is the ids of the 2d pts in frame after ransac filtering 
    for i, p in enumerate(pts_3d_euc):
        # the loop skips the current iteration and moves to the next point.
        pt = Point(mapp, p)
        pt.add_observation(currentFrame, idx[i])
        

    for pts_l in currentFrame.ptsLeft[idx]:
        u1, v1 = denormalize(K1, pts_l)
        cv2.circle(imgLeft, (u1,v1), 3, (0,255,0))

def process_frame(imgLeft, imgRight, prev_R=None, prev_t=None):
    imgLeft = cv2.resize(imgLeft, (W, H))
    imgRight = cv2.resize(imgRight, (W, H))
    currentFrame = StereoFrame(mapp, imgLeft, imgRight, K1, K2)

    if currentFrame.idLeft == 0:
        idxLeft, idxRight, R1, R2, t = matchStereoFrames(currentFrame)
        R_set = [R1, R1, R2, R2]  # Four rotation matrices from Essential Matrix decomposition
        t_set = [t, -t, t, -t]  # Four translation vectors from Essential Matrix decomposition
        baseline = np.linalg.norm(t)
        currentFrame.setInitialRightPose(baseline)
        R0 = np.eye(3)
        t0 = np.zeros((3,1))
        best_R, best_t, pts_3d_euc = triangulate_points_stereo(R0, t0, R_set, t_set, 
                                                       currentFrame.ptsLeft[idxLeft], 
                                                       currentFrame.ptsRight[idxRight],
                                                       first_frame=True)

        refined_pts_3d = refine_triangulation(
            pts_3d_euc,
            currentFrame.ptsLeft[idxLeft],
            currentFrame.ptsRight[idxRight],
            best_R, 
            best_t
        )

        # Optional: Filter out points with high reprojection error
        inlier_mask = filter_outliers(refined_pts_3d, 
                                    currentFrame.ptsLeft[idxLeft], 
                                    currentFrame.ptsRight[idxRight],
                                    best_R, best_t)
        pts_3d_euc = refined_pts_3d[inlier_mask]
        pose = np.eye(4)
        pose[:3, :3] = best_R
        pose[:3, 3] = best_t
        currentFrame.poseLeft = pose

        # Add points to the map
        print("left idx:", len(idxLeft))
        # add_to_map(imgLeft, currentFrame, pts_3d_euc, idxLeft)

        for i, p in enumerate(pts_3d_euc):
            #add 3d point p to map
            pt = Point(mapp, p)
            #add the current frame to the map
            #match idx of 2d pt index -> 3d point p id
            pt.add_observation(currentFrame, idxLeft[i])
        
        print(currentFrame.point_indices)

        for pts_l in currentFrame.ptsLeft[idxLeft]:
            u1, v1 = denormalize(K1, pts_l)
            cv2.circle(imgLeft, (u1,v1), 3, (0,255,0))

        print("pts 3d euc:", pts_3d_euc.shape)
        print("after adding the first frame, num 3d points in map:", len(mapp.points))
        return best_R, best_t
    
    #ALL GOOD TILL HERE
    else:
        # Find pose using RANSAC
        prevFrame = mapp.frames[-2]
        # print("currFrame id:", currentFrame.idLeft)
        # print("prev frame id:", prevFrame.idLeft)
        pts_3d = []
        pts_2d = []
        idxPrev, idxCurr = match_frames(prevFrame, currentFrame)

        for i, idx in enumerate(idxPrev):
            # Find corresponding 3D point
            if idx in list(prevFrame.point_indices.keys()):
                pt3d_idx = prevFrame.point_indices[idx]
                pts_3d.append(mapp.points[pt3d_idx].pt)
                pts_2d.append(currentFrame.ptsLeft[idxCurr[i]])

        print("3d - 2d:", len(pts_3d), len(pts_2d))

        if len(pts_3d) < 6:
            print("Not enough correspondences for PnP")
            exit()
            return prev_R, prev_t
        
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(np.array(pts_3d), 
                                                np.array(pts_2d),
                                                K1, None, reprojectionError=1.0)
    
        # Convert rotation vector to rotation matrix
        R_new, _ = cv2.Rodrigues(rvecs)
        t_new = tvecs
        R_new, t_new = nonlinear_pnp(np.array(pts_3d), np.array(pts_2d), K1, R_new, t_new)

        pose = np.eye(4)
        pose[:3, :3] = R_new
        pose[:3, 3] = t_new.reshape(3)
        currentFrame.poseLeft = pose

        # Triangulate new points from stereo pair
        idxLeft, idxRight, _, _, _ = matchStereoFrames(currentFrame)

        print("idxLeft:", len(idxLeft))
        print("idxRight", len(idxRight))

        # Filter out points that are already in the map
        new_idxLeft = []
        new_idxRight = []

        for i, (idxL, idxR) in enumerate(zip(idxLeft, idxRight)):
            if idxL not in list(currentFrame.point_indices.keys()):
                new_idxLeft.append(idxL)
                new_idxRight.append(idxR)

        print("new left", len(new_idxLeft))
        print("new right", len(new_idxRight))
        
        if len(new_idxLeft) > 0:
            # Triangulate new points
            
            # new_pts_3d = triangulate(
            #     K1, R_new, t_new, 
            #     np.array(currentFrame.ptsLeft)[new_idxLeft], 
            #     np.array(currentFrame.ptsRight)[new_idxRight]
            # )
            # new_pts_3d /= new_pts_3d[:, 3:]
            print(prev_R)
            print(prev_t)

            best_R, best_t, pts_3d_euc = triangulate_points_stereo(prev_R, prev_t, [R_new], [t_new], 
                                                       currentFrame.ptsLeft[idxLeft], 
                                                       currentFrame.ptsRight[idxRight],
                                                       first_frame=False)

            print("new 3d pts:", pts_3d_euc.shape)
            print("new 2d pts left:",np.array(currentFrame.ptsLeft)[new_idxLeft].shape)
            print("new 2d pts right:", np.array(currentFrame.ptsRight)[new_idxRight].shape)
            
            # Refine triangulation
            refined_new_pts = refine_triangulation(
                pts_3d_euc,
                np.array(currentFrame.ptsLeft)[new_idxLeft],
                np.array(currentFrame.ptsRight)[new_idxRight],
                R_new, t_new
            )
            
            # Filter outliers
            inlier_mask = filter_outliers(
                refined_new_pts,
                np.array(currentFrame.ptsLeft)[new_idxLeft],
                np.array(currentFrame.ptsRight)[new_idxRight],
                R_new, t_new
            )

            pts_3d_euc = refined_new_pts[inlier_mask]
            
            # Add new points to the map
            # add_to_map(
            #     imgLeft, 
            #     currentFrame, 
            #     refined_new_pts[inlier_mask], 
            #     np.array(new_idxLeft)[inlier_mask]
            # )

            for i, p in enumerate(pts_3d_euc):
                #add new3d points to the map
                pt = Point(mapp, p)
                #add 3d point observation to the current frame
                pt.add_observation(currentFrame, new_idxLeft[i])
                #add 3d point observation to the prev frame

            for pts_l in currentFrame.ptsLeft[new_idxLeft]:
                u1, v1 = denormalize(K1, pts_l)
                cv2.circle(imgLeft, (u1,v1), 3, (0,255,0))

        #2d display
        display.paint(imgLeft)
        #3d display
        mapp.display()
        return R_new, t_new

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

    calib_lines = read_calibration_file('dataset/sequences/00/calib.txt')
    intrinsic_matrix1 = extract_intrinsic_matrix(calib_lines, camera_id='P0')
    intrinsic_matrix2 = extract_intrinsic_matrix(calib_lines, camera_id='P1')

    img1 = cv2.imread("dataset/sequences/00/image_0/000000.png")
    height1, width1, channels1 = img1.shape
    print("Width:", width1)
    print("Height:", height1)
    
    if intrinsic_matrix1 is not None and intrinsic_matrix2 is not None:
        print("Intrinsic Matrix1 (K1):")
        print(intrinsic_matrix1)
        W = width1//2
        H = height1//2
        K1 = intrinsic_matrix1
        K1inv = np.linalg.inv(K1)

        print("Intrinsic Matrix2 (K2):")
        print(intrinsic_matrix2)
        K2 = intrinsic_matrix2
        K2inv = np.linalg.inv(K2)

    else:
        print("Intrinsic matrix not found for the specified camera ID.")
        #Default parameters:
        #define the principle point offset or the optical center coordinates
        W, H = 1920//2, 1080//2
        #define the focal length
        F=250
        #define the intrinsic matrix and the inverse of the intrinsic matrix
        K = np.array([[F, 0, W//2],
                    [0, F, H//2],
                    [0, 0, 1]])
        Kinv = np.linalg.inv(K)

    display = Display(W, H)
    mapp = Map()
    mapp.create_viewer()

    cap1 = cv2.VideoCapture("dataset/sequences/00/image_0/%06d.png")
    cap2 = cv2.VideoCapture("dataset/sequences/00/image_1/%06d.png")
    cap1.set(cv2.CAP_PROP_FPS, 10)  # Set to 10 FPS for testing
    cap2.set(cv2.CAP_PROP_FPS, 10)  # Set to 10 FPS for testing
    i = 0
    r, t = None, None
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 == True and ret2 == True:
            if i == 0:
                new_R, new_t = process_frame(frame1, frame2, r, t)
            else:
                new_R, new_t = process_frame(frame1, frame2, r, t)
            r = new_R
            t = new_t
            i+=1
        else:
            kill_all_processes_and_exit()