import numpy as np
import cv2

from lib.slam.feature_extraction import extract_features, match_features
from lib.slam.pose_estimation import estimate_pose
from lib.slam.triangulation import triangulate_points
from lib.evaluation.reprojection import compute_reproj_error
from lib.evaluation.trajectory_utils import smooth_trajectory

from lib.visualisation.opencv_vis import draw_keypoints, draw_matches, draw_reprojected_points

def pose_to_transform(R, t):
    """
    Convert relative rotation + translation to 4x4 matrix.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T

def run_slam(frames, K, visualize=True, max_jump_distance=1.0, smooth_alpha=0.5, skip_jumps=False):
    all_points = []
    detector = cv2.ORB_create(3000)
    prev_frame, prev_kp, prev_desc = None, None, None

    trajectory = []
    T_global = np.eye(4)
    jump_indices = []

    for i, frame in enumerate(frames):
        kp, desc = extract_features(frame, detector=detector)

        if prev_kp is not None:
            matches = match_features(prev_desc, desc)

            if len(matches) > 10:
                R, t, mask_pose = estimate_pose(prev_kp, kp, matches, K)
                if R is not None:
                    # Update global pose
                    T_rel = pose_to_transform(R, t)
                    T_global_new = T_global @ np.linalg.inv(T_rel)

                    # Detect jump
                    if skip_jumps:
                        dist = np.linalg.norm(T_global_new[:3,3] - T_global[:3,3])
                        if dist > max_jump_distance:
                            print(f"Jump detected at frame {i}, distance {dist:.2f}")
                            jump_indices.append(i)
                            # Skip triangulation for jump frames
                            T_global = T_global_new
                            trajectory.append(T_global.copy())
                            prev_frame, prev_kp, prev_desc = frame, kp, desc
                            continue

                    T_global = T_global_new
                    trajectory.append(T_global.copy())
                    pts3D = triangulate_points(prev_kp, kp, matches, K, R, t, mask_pose)

                    print(f"Frame {i}: Matches {len(matches)}, Inliers {np.sum(mask_pose)}, 3D pts {len(pts3D)}")
                    print("Camera pos:", T_global[:3,3])

                    # Evaluation
                    err = compute_reproj_error(pts3D, kp, matches, K, R, t, mask_pose)
                    if err is not None:
                        print(f"Reprojection error: {err:.2f} px")

                    if len(pts3D) > 0:
                        all_points.extend(pts3D)

                    # Visualization
                    if visualize:
                        key_img = draw_keypoints(frame, kp)
                        reproj_img = draw_reprojected_points(key_img, K, R, t, pts3D)
                        h, w = reproj_img.shape[:2]
                        scale = 640 / w
                        reproj_img = cv2.resize(reproj_img, (640, int(h * scale)))
                        cv2.imshow("SLAM Debug View", reproj_img)

        if visualize and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame, prev_kp, prev_desc = frame, kp, desc

    cv2.destroyAllWindows()

    # Smooth trajectory after processing all frames
    trajectory_smoothed = smooth_trajectory(trajectory, jump_indices=jump_indices, alpha=smooth_alpha)

    return np.array(all_points), trajectory_smoothed, jump_indices
