import math
import time
import cv2
import numpy as np
from pathlib import Path

def grab_keyframes(video_path, step=10, max_frames=200, min_sharpness=80.0, resize_max=1920):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames, kept = [], 0
    idx, total = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret = cap.grab()  # fast skip
        if not ret: break
        if idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret: break

            # optional downscale for speed (comment out to keep native res)
            h, w = frame.shape[:2]
            if max(h, w) > resize_max:
                scale = resize_max / max(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

            # sharpness filter (skip blurry)
            sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp >= min_sharpness:
                frames.append(frame)
                kept += 1
                if kept >= max_frames:
                    break
        idx += 1

    cap.release()
    if len(frames) < 2:
        raise RuntimeError("Not enough usable frames for stitching.")
    return frames

def stitch_frames(frames, mode="SCANS", out_path="mosaic.jpg"):
    mode_flag = cv2.Stitcher_SCANS if mode.upper()=="SCANS" else cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher_create(mode_flag)
    status, pano = stitcher.stitch(frames)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed, code={status}")
    cv2.imwrite(out_path, pano)
    return pano

if __name__ == "__main__":
    video_name = "cargo_ship.mp4"
    video_name_raw = Path(video_name).stem
    video = f"./images/{video_name}"
    
    time_start = time.time()
    frames = grab_keyframes(video, step=20, max_frames=1000, min_sharpness=100.0, resize_max=2048)
    time_end = time.time()
    print(f"Grabbing keyframes took: {math.floor((time_end - time_start) / 60)} minutes and {math.floor((time_end - time_start) % 60)} seconds")
    
    time_start = time.time()
    pano  = stitch_frames(frames, mode="SCANS", out_path=f"{video_name_raw}.jpg")
    time_end = time.time()
    print(f"Stitching frames took: {math.floor((time_end - time_start) / 60)} minutes and {math.floor((time_end - time_start) % 60)} seconds")
    print("Saved:", Path(f"{video_name_raw}.jpg").resolve())
