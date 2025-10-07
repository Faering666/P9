import cv2
import numpy as np
import time
import os

# =========================
# Configuration (edit these)
# =========================
SOURCE = "images/cargo_ship.mp4"              # 0 or "0" for webcam, "/path/to/video.mp4", "rtsp://...", "http://..."
SCALE = 0.5               # Downscale factor for speed (0.3–0.7 is typical)
PROCESS_EVERY = 2         # Process every Nth frame (higher = faster, less accurate)
PAD = 3.0                 # Panorama canvas size multiplier
RANSAC_THRESH = 3.0       # RANSAC reprojection threshold (pixels)
PACE_TO_FPS = True        # Keep playback roughly in real time for files/streams

# Save settings
SAVE_ON_EXIT = True       # Save panorama automatically when the program ends
OUTPUT_PATH = "./out/panorama.png"  # <-- change this to your desired path

# =========================
# Core helpers
# =========================
def make_canvas(h, w, pad=2.5):
    H, W = int(h * pad), int(w * pad)
    cy, cx = H // 2, W // 2
    return H, W, cy, cx

def to_gray_small(frame, scale=0.5):
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), frame

def blend_additive(canvas_acc, weight_acc, img, M):
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (canvas_acc.shape[1], canvas_acc.shape[0]))
    mask = cv2.warpPerspective(np.ones((h, w), dtype=np.float32), M, (canvas_acc.shape[1], canvas_acc.shape[0]))
    canvas_acc += warped.astype(np.float32)
    weight_acc += (mask[..., None])  # broadcast over 3 channels

def keypoints_and_desc(gray, orb):
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)
    return kp, des

def good_matches(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    goods = []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            goods.append(pair[0])
    return goods

def find_H(kp1, kp2, matches, ransac_thresh=3.0):
    if len(matches) < 8:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    return H

def open_capture(source):
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    try:
        idx = int(source)
        return cv2.VideoCapture(idx)
    except (ValueError, TypeError):
        return cv2.VideoCapture(source)

def finalize_panorama(canvas_acc, weight_acc):
    """Normalize and convert to 8-bit image."""
    pano = np.divide(canvas_acc, np.maximum(weight_acc, 1e-6))
    pano = np.clip(pano, 0, 255).astype(np.uint8)
    return pano

def save_panorama(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, image)
    if ok:
        print(f"[saved] {path}")
    else:
        print(f"[error] Could not save to {path}")

# =========================
# Runner
# =========================
def run_stitcher(
    source=SOURCE,
    scale=SCALE,
    process_every=PROCESS_EVERY,
    pad=PAD,
    ransac_thresh=RANSAC_THRESH,
    pace_to_fps=PACE_TO_FPS,
    save_on_exit=SAVE_ON_EXIT,
    output_path=OUTPUT_PATH,
):
    cap = open_capture(source)
    if not cap.isOpened():
        print(f"Failed to open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0
    frame_interval = 1.0 / fps

    ret, frame0 = cap.read()
    if not ret:
        print("Could not read the first frame.")
        cap.release()
        return

    gray0, small0 = to_gray_small(frame0, scale)
    h, w = small0.shape[:2]
    Hc, Wc, cy, cx = make_canvas(h, w, pad=pad)

    canvas_acc = np.zeros((Hc, Wc, 3), dtype=np.float32)
    weight_acc = np.zeros((Hc, Wc, 3), dtype=np.float32)

    orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=10)

    T0 = np.array([[1, 0, cx - w / 2],
                   [0, 1, cy - h / 2],
                   [0, 0, 1]], dtype=np.float32)
    blend_additive(canvas_acc, weight_acc, small0, T0)

    prev_kp, prev_des = keypoints_and_desc(gray0, orb)
    T_prev_to_canvas = T0.copy()

    i = 0
    last_time = time.perf_counter()
    last_pano = finalize_panorama(canvas_acc, weight_acc)

    while True:
        if pace_to_fps:
            now = time.perf_counter()
            dt = now - last_time
            if dt < frame_interval:
                time.sleep(frame_interval - dt)
            last_time = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        i += 1
        gray, small = to_gray_small(frame, scale)

        if i % process_every == 0:
            kp, des = keypoints_and_desc(gray, orb)
            if des is not None and prev_des is not None:
                matches = good_matches(des, prev_des, ratio=0.75)
                H_cur_to_prev = find_H(kp, prev_kp, matches, ransac_thresh=ransac_thresh)
                if H_cur_to_prev is not None:
                    T_cur_to_canvas = T_prev_to_canvas @ H_cur_to_prev
                    blend_additive(canvas_acc, weight_acc, small, T_cur_to_canvas)
                    prev_kp, prev_des = kp, des
                    T_prev_to_canvas = T_cur_to_canvas

        last_pano = finalize_panorama(canvas_acc, weight_acc)
        cv2.imshow("Panorama (q/ESC quit, r reset, s save)", last_pano)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        if key == ord('r'):
            canvas_acc[:] = 0
            weight_acc[:] = 0
            T0 = np.array([[1, 0, cx - w / 2],
                           [0, 1, cy - h / 2],
                           [0, 0, 1]], dtype=np.float32)
            blend_additive(canvas_acc, weight_acc, small, T0)
            prev_kp, prev_des = keypoints_and_desc(gray, orb)
            T_prev_to_canvas = T0.copy()
        if key == ord('s'):
            # manual save on demand
            save_panorama(last_pano, output_path)

    cap.release()
    cv2.destroyAllWindows()

    # Auto-save on exit if enabled
    if save_on_exit:
        save_panorama(last_pano, output_path)

# Run directly (or import and call run_stitcher with your own params)
if __name__ == "__main__":
    run_stitcher()
