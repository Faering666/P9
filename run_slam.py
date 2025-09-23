import numpy as np
import argparse

from lib.io.video_reader import read_video_frames
from lib.evaluation.trajectory_utils import detect_jumps, smooth_trajectory
from lib.visualisation.open3d_vis import visualize_trajectory_and_points 

from src.slam_pipeline import run_slam

def get_arguments():
    parser = argparse.ArgumentParser(prog="SLAM",
                                     description="Does the thing.",
                                     epilog="Hello Hugin")

    parser.add_argument("--video_path", type=str, help="Video path")
    parser.add_argument("--max_frames", type=int, help="Max frames", default=1000)
    parser.add_argument("--frame_skip", type=int, help="How much frame skip", default=1)
    parser.add_argument("--skip_jumps", type=bool, help="Should skip jumps", default=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()

    # Example intrinsics (fx, fy, cx, cy)
    fx, fy = 718.856, 718.856
    cx, cy = 607.1928, 185.2157
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    frames = read_video_frames(args.video_path, max_frames=args.max_frames, frame_skip=args.frame_skip)
    points, trajectory, jumps = run_slam(frames, K, skip_jumps=args.skip_jumps)

    print("Detected jumps at frames:", jumps)
    visualize_trajectory_and_points(points, trajectory)
