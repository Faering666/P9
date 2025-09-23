import numpy as np

def detect_jumps(trajectory, max_distance=1.0):
    """
    Detects jumps in a trajectory.
    
    trajectory: list of 4x4 np arrays
    max_distance: max allowed distance between consecutive poses
    
    Returns:
    - jump_indices: list of frame indices where a jump occurred
    """
    jump_indices = []
    for i in range(1, len(trajectory)):
        prev_pos = trajectory[i-1][:3, 3]
        curr_pos = trajectory[i][:3, 3]
        dist = np.linalg.norm(curr_pos - prev_pos)
        if dist > max_distance:
            jump_indices.append(i)
    return jump_indices

def smooth_trajectory(trajectory, jump_indices=None, alpha=0.5):
    """
    Smooth trajectory using simple exponential moving average.

    trajectory: list of 4x4 np arrays (camera poses)
    jump_indices: optional list of frame indices to ignore or skip smoothing
    alpha: smoothing factor (0-1), higher = more smoothing

    Returns:
        smoothed trajectory as list of 4x4 np arrays
    """
    smoothed = []
    for i, T in enumerate(trajectory):
        if i == 0:
            smoothed.append(T.copy())
        else:
            T_prev = smoothed[-1]
            # Optionally skip smoothing for jump frames
            if jump_indices is not None and i in jump_indices:
                smoothed.append(T.copy())
            else:
                pos_new = alpha * T[:3,3] + (1-alpha) * T_prev[:3,3]
                T_smooth = T.copy()
                T_smooth[:3,3] = pos_new
                smoothed.append(T_smooth)
    return smoothed