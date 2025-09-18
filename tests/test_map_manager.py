import pytest
import numpy as np
import os

from lib.map_manager import MapManager

test_map_file = "test_map.pcd"

@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(test_map_file):
        os.remove(test_map_file)
    yield
    if os.path.exists(test_map_file):
        os.remove(test_map_file)

def test_map_manager():
    manager = MapManager(map_file_path=test_map_file)

    initial_points = np.random.rand(1000, 3) * 10
    manager.add_points(initial_points)
    
    current_points_length = len(manager.point_cloud.points)
    assert current_points_length > 0
    
    new_points = np.random.rand(500, 3) * 5 + np.array([5, 5, 5])
    manager.add_points(new_points)
    assert len(manager.point_cloud.points) >= current_points_length

    drone_position = np.array([5.0, 5.0, 5.0])
    local_map_points = manager.query_local_map(drone_position, radius=3.0)
    assert local_map_points.shape[1] == 3
    assert local_map_points.shape[0] > 0
    
    print("Saving the final map to disk...")
    manager.save_to_disk()

    print("Loading the map from disk to verify...")
    new_manager = MapManager(map_file_path=test_map_file)
    print(f"Verified: The loaded map has {len(new_manager.point_cloud.points)} points.")
    assert len(new_manager.point_cloud.points) == len(manager.point_cloud.points)