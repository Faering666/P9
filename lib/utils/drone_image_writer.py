import json
from lib.data.drone_image_info import drone_image_info

def write_drone_image_info(drone_image_info):
    try: 
        with open('lib/data/drone_image_info.json', 'r+') as f:
            reader = json.load(f)
            reader["drone_image_info"].append({
                'position': drone_image_info.position,
                'angle': drone_image_info.angle,
                'rotation': drone_image_info.rotation,
                'timestamp': drone_image_info.timestamp,
                'image': drone_image_info.image
            })
            f.seek(0)
            json.dump(reader, f, indent=4)
            
    except FileNotFoundError:
        print("drone_image_info.json file not found.")
        return []