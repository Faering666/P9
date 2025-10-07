import math

from lib.drone.state import DroneState

def distance_meters(lat1, lon1, lat2, lon2):
    """
    Haversine formula for distance in meters.
    """
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

class PositionComparer:
    def __init__(self, pos_tol=2.0, yaw_tol=10.0, pitch_tol=3.0):
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol
        self.pitch_tol = pitch_tol

    def is_close_enough(self, current: DroneState, target: DroneState):
        dist = distance_meters(
            current.position[0], current.position[1],
            target.position[0], target.position[1]
        )

        alt_diff = abs(current.altitude - target.altitude)
        yaw_diff = abs(current.yaw - target.yaw)
        pitch_diff = abs(current.gimbal_pitch - target.gimbal_pitch)

        return (dist <= self.pos_tol and 
                alt_diff <= 1.0 and
                yaw_diff <= self.yaw_tol and
                pitch_diff <= self.pitch_tol)
