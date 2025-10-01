class poi
    def __init__(self, position, drone_images):
        self.position = position
        self.drone_images = drone_images

    def add_drone_image(self, drone_image):
        self.drone_images.append(drone_image)

    def info(self):
        return (self.position, [di.info() for di in self.drone_images])