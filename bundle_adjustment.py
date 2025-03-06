import gtsam
import numpy as np

def optimize(pose1, pose2, pts4d, keypoints_2d, K):
    # Convert homogeneous coordinates to Euclidean coordinates
    pts3d_euc = pts4d[:, :3] / pts4d[:, 3:]

    # Scale 3D points if necessary
    scale_factor = 0.1
    pts3d_euc = pts3d_euc * scale_factor
    
    # Convert intrinsic matrix to GTSAM format
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    gtsam_K = gtsam.Cal3_S2(fx, fy, 0, cx, cy)

    # Initialize factor graph and values
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

    # Adjust noise models
    pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, 2)
    point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 2)

    # Fix first camera to remove gauge freedom
    graph.add(gtsam.PriorFactorPose3(0, cam1_pose, pose_noise))

    for i, point in enumerate(pts3d_euc):
        point_key = i + 2
        initial_estimates.insert(point_key, gtsam.Point3(point))
        graph.add(gtsam.PriorFactorPoint3(point_key, gtsam.Point3(point), point_noise))

    # Normalize 2D keypoints
    def normalize_keypoints(keypoints, K):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        return [(((x - cx) / fx), ((y - cy) / fy)) for x, y in keypoints]

    normalized_keypoints = [normalize_keypoints(kps, K) for kps in keypoints_2d]

    # Add reprojection factors with a more robust cost function
    cauchy = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(1.0),
        gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
    )
    for cam_idx, keypoints in enumerate(normalized_keypoints):
        for i, (u, v) in enumerate(keypoints):
            point_key = i + 2
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(u, v), cauchy, cam_idx, point_key, gtsam_K))

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)  # Increase from default
    # params.setVerbosity("ERROR")  # or "TERMINATION" for more details
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
    
    print(f"Error before optimization: {optimizer.error()}")
    result = optimizer.optimize()
    print(f"Error after optimization: {optimizer.error()}")
    print(f"Number of iterations: {optimizer.iterations()}")

    # Extract results
    optimized_poses = [result.atPose3(i) for i in range(2)]
    optimized_points = np.array([result.atPoint3(2 + i) for i in range(len(pts4d))]) // scale_factor

    return optimized_poses, optimized_points