# World-Space Hand Motion Reconstruction

This project provides a "DIY" pipeline to extract 3D hand keypoints from a moving camera (egocentric videos) and transform them into a stable world coordinate system. 

It accomplishes this by stitching together:
1. **MediaPipe Hands:** For fast, robust 2D tracking of the hand.
2. **Transformers (Depth Anything / MiDaS):** For dense depth estimation of the scene.
3. **Dummy SLAM Interface:** A mock generator representing a SLAM system providing Camera-to-World extrinsic matrices (you can plug in DROID-SLAM or ORB-SLAM3 here).

## Prerequisites
- Python 3.8+
- Webcam (for live demo) or a video file.

## Installation

1. Clone or download this repository.
2. Install the necessary pip dependencies:
```bash
pip install -r requirements.txt
```

*(Note: The first time you run the script, HuggingFace's `transformers` library will download the `Intel/dpt-large` MiDaS model, which is ~1.4GB.)*

## How it Works: The Math

The pipeline essentially performs a **Local to Global** 3D coordinate transformation.

### 1. 2D to 3D Camera Space (Pinhole Model)
First, MediaPipe extracts the 2D pixel coordinates $(u, v)$ of the hand root (the wrist). The depth model predicts a depth map. We look up the depth $Z$ at $(u, v)$.
Using the intrinsic parameters of the camera (focal length $f_x, f_y$ and principal point $c_x, c_y$), we unproject the pixel into 3D metric camera space $(X_c, Y_c, Z_c)$:

$$X_c = (u - c_x) \cdot \frac{Z}{f_x}$$
$$Y_c = (v - c_y) \cdot \frac{Z}{f_y}$$

### 2. Camera Space to World Space
The SLAM algorithm provides the instantaneous 3D pose of the camera in the fixed world view, represented as a rotation matrix $R_{c2w}$ (3x3) and a translation vector $t_{c2w}$ (3x1).

To find the hand's absolute world position $P_{world}$, we apply the extrinsic transformation to the local hand position $P_{camera} = [X_c, Y_c, Z_c]^T$:

$$P_{world} = R_{c2w} \cdot P_{camera} + t_{c2w}$$

This isolates the hand's true motion from the motion of the user's head/camera.

## Running the Pipeline

### Step 1: Run the Tracker
Run the main script. If you want to test the math without a webcam, use the `--dummy` flag. It will run 30 frames of simulated motion and tracking.

```bash
# Run with Webcam
python core_pipeline.py

# OR Run with Dummy data for quick verification
python core_pipeline.py --dummy --frames 50
```
This script will produce a `trajectory_data.json` file logging the absolute world coordinates.

### Step 2: Visualize the 3D Trajectory
Once `trajectory_data.json` is generated, run the visualization script.

```bash
python visualization.py
```
This will open an interactive 3D matplotlib plot showing the camera moving over time (blue) relative to the hand position (red). It will also save a picture `trajectory_plot.png`.
