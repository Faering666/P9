import numpy as np

from lib.io.video_reader import read_video_frames
from lib.evaluation.trajectory_utils import detect_jumps, smooth_trajectory
from lib.visualisation.open3d_vis import visualize_trajectory_and_points 

from src.slam_pipeline import run_slam

if __name__ == "__main__":
    video_path = "data/city.mp4"

    # Example intrinsics (fx, fy, cx, cy)
    fx, fy = 718.856, 718.856
    cx, cy = 607.1928, 185.2157
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # frames = read_video_frames(video_path, max_frames=300, frame_skip=2)
    frames = read_video_frames(video_path, max_frames=200)
    points, trajectory, jumps = run_slam(frames, K)

    print("Detected jumps at frames:", jumps)
    visualize_trajectory_and_points(points, trajectory)
