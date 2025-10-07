import cv2
from lib.io.video_feed import VideoFeed

from lib.detection.img_comparer import ImageComparer

from lib.utils.drone_image_loader import load_drone_image_info
from lib.utils.drone_image_writer import write_drone_image_info

from lib.data.drone_image_info import drone_image_info
from lib.data.poi import poi

from lib.drone.state import DroneState
from lib.drone.position_comparer import PositionComparer


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

def run_add_drone_image_demo():
    dii = drone_image_info((1,2), 45, 90, "2023-10-01T12:00:00Z", "image_003.png")
    write_drone_image_info(dii)

def run_position_demo():
    target = DroneState((55.0, 12.0), 50.0, yaw=90, gimbal_pitch=-30, gimbal_yaw=0)
    current = DroneState((55.00001, 12.00001), 50.5, yaw=92, gimbal_pitch=-31, gimbal_yaw=0)

    position_comparer = PositionComparer()
    print("Target:", target.info())
    print("Current:", current.info())
    print("Close enough:", position_comparer.is_close_enough(current, target))

if __name__ == "__main__":
    run_demo()
    run_drone_data_demo()
    run_add_drone_image_demo()
    run_position_demo()
