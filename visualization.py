import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def render_trajectory(json_path, save_path="trajectory_plot.png"):
    print(f"Loading trajectory data from {json_path}...")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}. Please run core_pipeline.py first.")
        return

    cameras = np.array(data.get("camera_positions", []))
    hands = np.array(data.get("hand_positions", []))

    if len(cameras) == 0:
        print("No camera trajectory data found in the file.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera trajectory (Blue)
    ax.plot(cameras[:, 0], cameras[:, 1], cameras[:, 2], 
            label='Camera Trajectory', color='blue', linewidth=2, marker='o', markersize=3)
    
    # Plot hand trajectory (Red)
    if len(hands) > 0:
        ax.plot(hands[:, 0], hands[:, 1], hands[:, 2], 
                label='Hand Trajectory', color='red', linewidth=2, marker='^', markersize=3)
        
        # Draw lines connecting the camera to the hand at each frame
        # Make sure arrays are same length (they usually are, but just in case)
        min_len = min(len(cameras), len(hands))
        for i in range(min_len):
            ax.plot([cameras[i, 0], hands[i, 0]], 
                    [cameras[i, 1], hands[i, 1]], 
                    [cameras[i, 2], hands[i, 2]], 
                    color='gray', linestyle=':', alpha=0.3)

    ax.set_title('3D World-Space Trajectory')
    ax.set_xlabel('X (World)')
    ax.set_ylabel('Y (World)')
    ax.set_zlabel('Z (World)')
    ax.legend()

    # Equalize axis aspect ratio for better visualization
    all_points = np.vstack([cameras, hands]) if len(hands) > 0 else cameras
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    plt.tight_layout()
    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path, dpi=300)
    print("Showing interactive plot. Close the window to exit.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the 3D trajectory of the camera and hand.")
    parser.add_argument("--input", type=str, default="trajectory_data.json", help="Path to the JSON trajectory data.")
    parser.add_argument("--output", type=str, default="trajectory_plot.png", help="Path to save the output plot.")
    args = parser.parse_args()

    render_trajectory(args.input, args.output)
