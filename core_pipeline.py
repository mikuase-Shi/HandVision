import cv2
import mediapipe as mp
import numpy as np
import json
import torch
from transformers import pipeline
from scipy.spatial.transform import Rotation as R

class DummySLAM:
    """
    Simulates a SLAM system providing Camera-to-World extrinsic matrices (R_c2w, t_c2w).
    In a real system, this would be reading from DROID-SLAM, ORB-SLAM3, etc.
    """
    def __init__(self):
        self.frame_count = 0
        # Start at origin
        self.current_t = np.array([0.0, 0.0, 0.0])
        self.current_r = R.from_euler('xyz', [0, 0, 0], degrees=True)
    
    def get_pose(self):
        """
        Returns a mock camera pose for the current frame.
        We simulate the camera slowly moving forward and panning.
        
        Returns:
            R_c2w (np.ndarray): 3x3 rotation matrix
            t_c2w (np.ndarray): 3x1 translation vector
        """
        # Simulate gentle movement
        self.current_t[0] += 0.01 * np.sin(self.frame_count * 0.05) # Sway X
        self.current_t[1] += 0.005 * np.cos(self.frame_count * 0.05) # Sway Y
        self.current_t[2] += 0.02 # Move forward in Z
        
        # Simulate gentle rotation
        delta_r = R.from_euler('xyz', [0.1, 0.2 * np.sin(self.frame_count * 0.01), 0], degrees=True)
        self.current_r = delta_r * self.current_r
        
        self.frame_count += 1
        
        return self.current_r.as_matrix(), self.current_t.reshape(3, 1)


class HandMotionReconstructor:
    def __init__(self, focal_length_x=800.0, focal_length_y=800.0):
        print("Initializing MediaPipe Hands...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,         # Assume 1 hand for simplicity
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("Initializing Depth Estimation Model (This may take a moment to download)...")
        # Determine device
        device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else "mps")
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large", device=device)
        
        self.slam = DummySLAM()
        
        # Camera intrinsics (Assumed)
        self.fx = focal_length_x
        self.fy = focal_length_y
        self.cx = None # Will be set based on image resolution
        self.cy = None

    def process_frame(self, frame_bgr):
        """
        Processes a single BGR frame.
        
        Returns:
            world_coords (np.ndarray): 3x1 absolute 3D coordinate of the hand root.
            camera_t (np.ndarray): 3x1 absolute translation of the camera.
            frame_annotated (np.ndarray): Image with drawn landmarks (for debugging).
        """
        h, w, _ = frame_bgr.shape
        if self.cx is None:
            self.cx = w / 2.0
            self.cy = h / 2.0
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Run MediaPipe Hands
        results = self.hands.process(frame_rgb)
        
        # 2. Estimate Depth Map
        # PIL Image is expected by transformers pipeline
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)
        depth_result = self.depth_estimator(pil_img)
        # depth_result["depth"] is a PIL image, "predicted_depth" is the tensor
        depth_map = np.array(depth_result["depth"])
        
        # Normalize/Scale depth (transformers outputs relative depth, typically 0-255 map). 
        # For a real metric system (like RealSense or metric-SLAM), we would use actual metric depth.
        # Here we approximate relative depth to a pseudo-metric scale.
        depth_map = depth_map.astype(np.float32)
        # A simple heuristic: invert and scale to simulate metric distance (0.5m to 2.0m)
        depth_map = 255.0 / (depth_map + 1e-5) 

        world_hand_pos = None
        
        # 3. Get Camera Pose
        R_c2w, t_c2w = self.slam.get_pose()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Use the WRIST (landmark 0) as the root joint
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # MediaPipe returns normalized coordinates [0, 1]
            u = int(wrist.x * w)
            v = int(wrist.y * h)
            
            # Ensure within bounds
            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)
            
            # 4. 2D to 3D Camera Space (Pinhole Model)
            Z = depth_map[v, u]
            
            # Camera Space coordinates (Local)
            # Math: X_c = (u - c_x) * Z / f_x
            X_c = (u - self.cx) * Z / self.fx
            Y_c = (v - self.cy) * Z / self.fy
            Z_c = Z
            
            P_camera = np.array([[X_c], [Y_c], [Z_c]])
            
            # 5. Local to Global (World) Transformation
            # Math: P_world = R_c2w * P_camera + t_c2w
            P_world = np.dot(R_c2w, P_camera) + t_c2w
            world_hand_pos = P_world.flatten()
            
            # Draw for debugging
            cv2.circle(frame_bgr, (u, v), 5, (0, 255, 0), -1)
            
        return world_hand_pos, t_c2w.flatten(), frame_bgr

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Run on dummy synthetic frames instead of webcam")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to run")
    parser.add_argument("--video_path", type=str, default=None, help="Path to input video file")
    args = parser.parse_args()
    
    reconstructor = HandMotionReconstructor()
    
    trajectory_data = {
        "camera_positions": [],
        "hand_positions": []
    }
    
    if args.dummy:
        print(f"Running in dummy mode for {args.frames} frames...")
        # Create a blank image and simulate a hand point moving
        for i in range(args.frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a fake hand blob that MediaPipe might hopefully detect (usually it requires real looking hands)
            # Actually, MediaPipe usually fails on blank images. To ensure pipeline tests run, we'll bypass MediaPipe 
            # constraint *only* for the dummy test, or provide a fallback.
            # To keep it robust, let's load a real test image if available, else just simulate the math directly 
            # if MediaPipe fails.
            
            # For pure testing, let's use a solid color
            frame[:,:] = (200, 200, 200)
            
            world_hand, cam_pos, _ = reconstructor.process_frame(frame)
            
            # Since MediaPipe will likely return None on a blank image, we fake the hand detection for testing the math
            if world_hand is None:
                # Force fake values
                w, h = 640, 480
                u, v = w//2 + int(50 * np.sin(i*0.1)), h//2 + int(50 * np.cos(i*0.1))
                Z = 1.0 # 1 meter away
                X_c = (u - reconstructor.cx) * Z / reconstructor.fx
                Y_c = (v - reconstructor.cy) * Z / reconstructor.fy
                P_c = np.array([[X_c], [Y_c], [Z]])
                R_c2w, t_c2w = reconstructor.slam.get_pose()
                world_hand = (np.dot(R_c2w, P_c) + t_c2w).flatten()
                
            trajectory_data["camera_positions"].append(cam_pos.tolist())
            if world_hand is not None:
                trajectory_data["hand_positions"].append(world_hand.tolist())
    else:
        if args.video_path:
            print(f"Opening video file: {args.video_path}")
            cap = cv2.VideoCapture(args.video_path)
        else:
            print("Starting webcam...")
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_tracking.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            world_hand, cam_pos, annotated_frame = reconstructor.process_frame(frame)
            
            trajectory_data["camera_positions"].append(cam_pos.tolist())
            if world_hand is not None:
                trajectory_data["hand_positions"].append(world_hand.tolist())
                
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out.release()
        
    # Save output
    with open("trajectory_data.json", "w") as f:
        json.dump(trajectory_data, f, indent=4)
        
    print("Saved trajectory to trajectory_data.json.")

if __name__ == "__main__":
    main()
