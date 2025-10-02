class DroneState:
    def __init__(self, position, altitude, yaw, gimbal_pitch, gimbal_yaw):
        """
        position: tuple (lat, lon)
        altitude: float (meters)
        yaw: float (degrees, 0-360, compass heading)
        gimbal_pitch: float (degrees, camera tilt)
        gimbal_yaw: float (degrees, camera pan relative to drone)
        """
        self.position = position
        self.altitude = altitude
        self.yaw = yaw
        self.gimbal_pitch = gimbal_pitch
        self.gimbal_yaw = gimbal_yaw

    def info(self):
        return {
            "position": self.position,
            "altitude": self.altitude,
            "yaw": self.yaw,
            "gimbal_pitch": self.gimbal_pitch,
            "gimbal_yaw": self.gimbal_yaw,
        }

