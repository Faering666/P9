import open3d as o3d
import numpy as np

def visualize_trajectory_and_points(points, trajectory, draw_axes_every=10):
    vis_elems = []

    if len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        vis_elems.append(pcd)

    if len(trajectory) > 1:
        positions = np.array([T[:3, 3] for T in trajectory])
        lines = [[i, i+1] for i in range(len(positions)-1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # red
        vis_elems.append(line_set)

        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        start_sphere.paint_uniform_color([0, 1, 0])  # green
        start_sphere.translate(positions[0])
        vis_elems.append(start_sphere)

        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        end_sphere.paint_uniform_color([0, 0, 1])  # blue
        end_sphere.translate(positions[-1])
        vis_elems.append(end_sphere)

        for i, T in enumerate(trajectory):
            if i % draw_axes_every == 0 or i == 0 or i == len(trajectory)-1:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                axis.transform(T)
                vis_elems.append(axis)

    o3d.visualization.draw_geometries(vis_elems)
