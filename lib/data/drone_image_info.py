class drone_image_info:
    def __init__(self, position, angle, rotation, timestamp):
        self.position = position
        self.angle = angle
        self.rotation = rotation
        self.timestamp = timestamp


    def info(self):
        return (self.position, self.angle, self.rotation, self.timestamp)