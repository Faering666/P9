import cv2
from lib.io.video_feed import VideoFeed
from lib.detection.img_comparer import ImageComparer
from lib.utils.drone_image_loader import load_drone_image_info
from lib.data.drone_image_info import drone_image_info
from lib.data.poi import poi

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

def run_drone_data_demo():
    data = load_drone_image_info()
    p = poi((0,0), [])
    for entry in data:
        dii = drone_image_info(entry[0], entry[1], entry[2], entry[3], entry[4])
        p.add_drone_image(dii)
    
    print(p.info())




if __name__ == "__main__":
    # run_demo()
    run_drone_data_demo()