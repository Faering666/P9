import open3d as o3d
import numpy as np

import os

class MapManager:
    def __init__(self, map_file_path="slam_map.pcd"):
        self.map_file_path = map_file_path
        self.point_cloud = o3d.geometry.PointCloud()

        if os.path.exists(self.map_file_path):
            self.load_from_disk()

    def add_points(self, new_points: np.ndarray):
        """
        Adds new points (from SLAM algorithm) to the in-memory point cloud.

        Args:
            new_points: A numpy array of shape (N, 3) representing new 3D points.
        """
        if new_points.shape[0] == 0:
            return
        
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)

        # Merge points with existing map
        self.point_cloud += new_pcd

        # Downsample point cloud to keep manageable
        self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size=0.1)

        print(f"Added {new_points.shape[0]} points. Total points: {len(self.point_cloud.points)}")

    def query_local_map(self, drone_position: np.ndarray, radius: float=5.0):
        """
        Performs a fast spatial query to get points within a radius of the drone's position.

        Args:
            drone_position: The (x, y, z) coordinates of the drone as a numpy array.
            radius: The radius of the query sphere.

        Returns:
            A numpy array of points within the specified radius.
        """
        if not self.point_cloud.points:
            return np.array([])

        kdtree = o3d.geometry.KDTreeFlann(self.point_cloud)

        [k, idx, _] = kdtree.search_radius_vector_3d(drone_position, radius)

        local_points = np.asarray(self.point_cloud.points)[idx, :]
        return local_points

    def save_to_disk(self):
        """
        Saves the entire point cloud to disk in a standard format (.pcd).
        """
        o3d.io.write_point_cloud(self.map_file_path, self.point_cloud)
        print(f"Map saved to {self.map_file_path}")

    def load_from_disk(self):
        """
        Loads a point cloud from disk and initialises the internal map.
        """
        try:
            self.point_cloud = o3d.io.read_point_cloud(self.map_file_path)
            print(f"Map loaded from {self.map_file_path} with {len(self.point_cloud.points)} points.")
        except Exception as e:
            print(f"Error loading map: {e}")
            self.point_cloud = o3d.geometry.PointCloud()
