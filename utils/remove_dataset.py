import os

root_dir = "data/"
for image_name in os.listdir(root_dir):
    for x in [0, 1, 2, 3]:
        if image_name.endswith(f'_{x}.jpg'):
            print(image_name)
            os.remove(os.path.join(root_dir, image_name))
