import cv2
import numpy as np
import gtsam
from pointmap import Map, Point
from display import Display
from utils import add_ones, triangulate, pose3_to_matrix
from extractor import Frame, match_frames, denormalize
from multiprocessing import set_start_method, active_children
from bundle_adjustment import optimize

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

    # print("before optimization:\n")
    # print("pts4d:",pts4d[0:5])
    # print("euclidean:")
    # temp = pts4d / pts4d[:, 3:]
    # print(temp[0:5])
    # print("pose1")
    # print(f1.pose)
    # print("pose2")
    # print(f2.pose)

    # Run Bundle Adjustment
    optimized_poses, optimized_points = optimize(f1.pose, f2.pose, pts4d, [f2.pts[idx2], f1.pts[idx1]], K)

    # Update camera poses and 3D points with optimized values
    # Convert optimized poses back to 4x4 transformation matrices
    f1.pose = pose3_to_matrix(optimized_poses[0])
    f2.pose = pose3_to_matrix(optimized_poses[1])

    # print("After optimization:\n", optimized_points[0:5])

    pts4d = np.hstack((optimized_points, np.ones((optimized_points.shape[0], 1))))

    # print(pts4d[0:5])
    # print(f1.pose, f2.pose)

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

    #2d display
    display.paint(img)

    #3d display
    mapp.display()

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

    cap = cv2.VideoCapture("videos/test_countryroad.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            kill_all_processes_and_exit()