from dataclasses import dataclass

@dataclass
class DroneState:
    latitude: float
    longitude: float
    altitude: float
    yaw: float              # heading (degrees)
    gimbal_pitch: float     # tilt
    gimbal_yaw: float       # pan

