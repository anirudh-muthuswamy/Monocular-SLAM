# Monocular SLAM

A lightweight monocular SLAM system that extracts ORB features, estimates camera motion, and reconstructs sparse 3D points. Future work includes pose graph optimization and loop closure detection.

## Features

- **Feature Extraction & Matching**

  - Extract ORB features and compute descriptors
  - Match features using KNN with Hamming distance
  - Filter outliers using Lowe’s ratio test and RANSAC

- **Camera Motion Estimation**

  - Compute fundamental matrix using the 8-point algorithm
  - Decompose fundamental matrix using SVD to obtain the essential matrix
  - Extract camera pose (rotation & translation) from the essential matrix

- **3D Reconstruction**

  - Triangulate feature points into 3D homogeneous coordinates
  - Visualize 2D feature tracks using OpenCV
  - Visualize 3D points and camera poses using VTK

- **Future Enhancements**

  - Implement pose graph optimization / bundle adjustment using GTSAM
  - Add loop closure detection using Bag of Words

## Requirements

Install the required dependencies before running the project:

```bash
pip install numpy opencv-python gtsam vtk scikit-image pysdl2 pysdl2-dll
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/monocular-slam.git
   cd monocular-slam
   ```
2. Run the main script:
   ```bash
   python main.py
   ```

### Acknowledgments

Big thanks to https://github.com/geohot for the inspiration. This project builds upon ideas and implementations from their work.