class drone_image:
    def __init__(self, position, angle):
        self.position = position
        self.angle = angle


    def info(self):
        return (self.position, self.angle)