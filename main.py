import cv2
import numpy as np
from pointmap import Map, Point
from display import Display
from extractor import Frame, match_frames, add_ones, denormalize
from multiprocessing import set_start_method, active_children
from utils import extract_intrinsic_matrix, read_calibration_file

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

    #homoegenous coordinates [x,y,z,w] are converted to euclidean coordinates
    pts4d /= pts4d[:, 3:]

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

    calib_file_path = "dataset/sequences/00/calib.txt"
    calib_lines = read_calibration_file(calib_file_path)
    intrinsic_matrix = extract_intrinsic_matrix(calib_lines, camera_id='P0')
    
    img = cv2.imread("dataset/sequences/00/image_0/000000.png")
    height, width = img.shape[:2]  # Get first two elements regardless of channels
    print("Width:", width)
    print("Height:", height)
    W = width
    H = height
    if intrinsic_matrix is not None:
        print("Intrinsic Matrix (K):")
        print(intrinsic_matrix)
        K = intrinsic_matrix
        Kinv = np.linalg.inv(K)
    else:
        print("Intrinsic matrix not found for the specified camera ID.")
        #Default parameters
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

    # cap = cv2.VideoCapture("videos/test_countryroad.mp4")
    cap = cv2.VideoCapture("dataset/sequences/00/image_0/%06d.png")
    cap.set(cv2.CAP_PROP_FPS, 10)  # Set to 10 FPS for testing
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            kill_all_processes_and_exit()