import cv2

def read_video_frames(video_path, max_frames=None, frame_skip=1):
    """
    Yields frames from video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            yield frame

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
