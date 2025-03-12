import numpy as np
from scipy.optimize import least_squares

def refine_triangulation(pts_3d_euc, pts_left_norm, pts_right_norm, best_R, best_t):

    # Left camera pose (identity in your setup)
    R_left = np.eye(3)
    t_left = np.zeros(3)
    
    # Right camera pose
    R_right = best_R
    t_right = best_t
    
    # Define cost function for optimization
    def cost_function(points_vector):
        # Reshape vector to Nx3 array of 3D points
        points_3d = points_vector.reshape(-1, 3)
        
        # Project points to left camera (normalized coordinates)
        projected_left = project_points(points_3d, R_left, t_left)
        
        # Project points to right camera (normalized coordinates)
        projected_right = project_points(points_3d, R_right, t_right)
        
        # Compute reprojection errors
        errors_left = (projected_left - pts_left_norm).flatten()
        errors_right = (projected_right - pts_right_norm).flatten()
        
        # Combine errors from both cameras
        return np.concatenate([errors_left, errors_right])
    
    # Run optimization using Levenberg-Marquardt algorithm
    result = least_squares(cost_function, pts_3d_euc.flatten(), method='lm', verbose=1)
    
    # Reshape result back to Nx3 array
    refined_points_3d = result.x.reshape(-1, 3)
    
    return refined_points_3d

def project_points(points_3d, R, t):
    """Project 3D points to normalized image coordinates"""
    # Convert to homogeneous coordinates
    n_points = points_3d.shape[0]
    points_3d_h = np.hstack([points_3d, np.ones((n_points, 1))])
    
    # Create transformation matrix
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t.flatten()
    
    # Transform points to camera coordinates
    points_camera = points_3d_h @ Rt.T
    
    # Perform perspective division
    points_camera_normalized = points_camera[:, :3] / points_camera[:, 2:3]
    
    # Return x, y coordinates (normalized)
    return points_camera_normalized[:, :2]

def filter_outliers(points_3d, pts_left_norm, pts_right_norm, best_R, best_t, threshold=0.01):

    # Project points to both cameras
    proj_left = project_points(points_3d, np.eye(3), np.zeros(3))
    proj_right = project_points(points_3d, best_R, best_t)
    
    # Calculate reprojection errors
    error_left = np.sqrt(np.sum((proj_left - pts_left_norm)**2, axis=1))
    error_right = np.sqrt(np.sum((proj_right - pts_right_norm)**2, axis=1))
    
    # Average error across both cameras
    avg_error = (error_left + error_right) / 2

    # Return mask of inliers
    return avg_error < threshold
