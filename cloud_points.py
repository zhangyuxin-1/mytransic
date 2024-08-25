import pyrealsense2 as rs
import numpy as np
import cv2
import pupil_apriltags
# import apriltag
import open3d as o3d
from eye_to_hand import *


def detect_apriltag():
    # 初始化RealSense相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    pipeline.start(config)

    # 创建AprilTag检测器
    detector = pupil_apriltags.Detector()

    all_point_camera = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 将图像转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            # print(color_image)
            depth_image = np.asanyarray(depth_frame.get_data())

            # 将彩色图像转换为灰度图像
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 检测AprilTag
            tags = detector.detect(gray_image)

            for tag in tags:
                # 绘制AprilTag边框
                for idx in range(len(tag.corners)):
                    pt1 = (int(tag.corners[idx][0]), int(tag.corners[idx][1]))
                    pt2 = (int(tag.corners[(idx + 1) % len(tag.corners)][0]), int(tag.corners[(idx + 1) % len(tag.corners)][1]))
                    cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

                # 获取AprilTag中心点的像素坐标
                center_x, center_y = int(tag.center[0]), int(tag.center[1])

                # 获取该点的深度值
                depth_value = depth_image[center_y, center_x]

                # 将像素坐标转换为相机坐标（以米为单位）
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point_camera = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth_value)

                # collect the point_camera
                
                all_point_camera.append(point_camera)
                # 打印AprilTag的相机坐标系下的坐标
                print("AprilTag Camera Coordinates: ", point_camera)

            # 显示结果图像
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        all_point_camera = np.array(all_point_camera)
    return all_point_camera


if __name__ == "__main__":
 # 假设你有AprilTag中心点的坐标列表 (N x 3 的numpy数组)

    external_image_path = '/home/zhangyuxin/furniture-bench-main/save_chessboard/test_camera_external'
    chess_board_x_num = 11
    chess_board_y_num = 8
    chess_board_len = 20.0
     #相机内参，需要根据实际内参大小进行修改
    fx = 4.769347547203772137e+02
    fy = 4.776687745938567673e+02
    cx = 3.088977531883862753e+02
    cy = 1.795221462104951229e+02
    
    K=np.array([[fx,0,cx],
                [0,fy,cy],
                [0,0,1]],dtype=np.float64)
    
    all_point_camera = detect_apriltag()
    # print(all_point_camera.shape)
    # 创建一个Open3D的点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 将坐标设置为点云的点
    point_cloud.points = o3d.utility.Vector3dVector(all_point_camera)

    o3d.visualization.draw_geometries([point_cloud])
    all_point_camera = np.hstack((all_point_camera, np.ones((all_point_camera.shape[0], 1))))
    
    transformation_matrix = cam_to_world(external_image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)

    world_points_hom = (transformation_matrix @ all_point_camera.T).T

    world_points = world_points_hom[:, :3]
    # print(world_points)

    
    
    # # 保存点云为 .ply 文件
    # o3d.io.write_point_cloud("april_tag_point_cloud.ply", point_cloud)

    # # 或者保存为 .pcd 文件
    # o3d.io.write_point_cloud("april_tag_point_cloud.pcd", point_cloud)

    # print("点云已保存")