import cv2

class VideoFeed:
    """
    Handles video input from webcam, file, or network stream.
    """

    def __init__(self, source=0):
        """
        Args:
            source (int or str): 
                - int: webcam index (default 0)
                - str: path to video file (e.g., "video.mp4")
                - str: network stream URL (e.g., "rtsp://...")
        """
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

    def get_frame(self):
        """
        Returns the next frame from the video source.
        Returns:
            frame (ndarray) or None if video ended.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Releases the video source."""
        self.cap.release()
