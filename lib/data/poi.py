class poi:
    def __init__(self, position, drone_images_info):
        self.position = position
        self.drone_images_info = drone_images_info 

    def add_drone_image(self, drone_image_info):
        self.drone_images_info.append(drone_image_info)
    
    def add_position(self, position):
        self.position = position

    def info(self):
        return (self.position, [dii.info() for dii in self.drone_images_info])