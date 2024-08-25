import open3d as o3d
import numpy as np

# 创建一个Open3D的点云对象
point_cloud = o3d.geometry.PointCloud()

# 将坐标设置为点云的点
point_cloud.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))

o3d.visualization.draw_geometries([point_cloud])