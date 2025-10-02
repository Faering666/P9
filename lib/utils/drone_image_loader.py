import json

def load_drone_image_info():
    try: 
        with open('lib/data/drone_image_info.json', 'r') as f:
            reader = json.load(f)
            data = [(
                entry['position'], 
                entry['angle'], 
                entry['rotation'], 
                entry['timestamp'], 
                entry['image']
                ) for entry in reader["drone_image_info"]]

            return data
            
    except FileNotFoundError:
        print("drone_image_info.json file not found.")
        return []