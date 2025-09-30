import cv2
from lib.io.video_feed import VideoFeed
from lib.detection.img_comparer import ImageComparer

def run_demo():
    feed = VideoFeed("data/video.mp4")
    comparer = ImageComparer()

    reference = cv2.imread("data/reference.png")

    while True:
        frame = feed.get_frame()
        if frame is None:
            break

        sim = comparer.compare(frame, reference, visualize=True)
        print(f"Similarity: {sim:.2f}")

    feed.release()

if __name__ == "__main__":
    run_demo()
