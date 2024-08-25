import os
import pyrealsense2 as rs
import cv2
import numpy as np
from PIL import Image
import math
from franka_pose import *
import pandas as pd
import csv
import glob


def get_name(floder_path):
    image_path = glob.glob(floder_path)
    image_name = [os.path.basename(file) for file in image_path]
    return image_name

def get_RT_from_chessboard(external_image_path, chess_board_x_num, chess_board_y_num, chess_board_len, K):
    images = read_images_from_folder(external_image_path)
    
    # 阈值
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # print(cv2.TERM_CRITERIA_EPS,'',cv2.TERM_CRITERIA_MAX_ITER)
    save_RT = []
    # 遍历所有标定图像
    for image in images:
        
        image = cv2.imread(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None)
           
        # 如果找到棋盘格角点则存储对应的3D和2D坐标
        if ret:
            # 精细查找角点
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(image, (chess_board_x_num, chess_board_y_num), corners2, ret)  #调换以后方向才对

            corner_points=np.zeros((2,corners.shape[0]),dtype=np.float64)
            for i in range(corners.shape[0]):
                corner_points[:,i]=corners[i,0,:]
            
            object_points=np.zeros((3,chess_board_x_num*chess_board_y_num),dtype=np.float64)
            flag = 0
            for i in range(chess_board_y_num):
                for j in range(chess_board_x_num):
                    object_points[:2,flag]=np.array([(11-j-1)*chess_board_len,(8-i-1)*chess_board_len])
                    flag+=1
            
            retval,rvec,tvec  = cv2.solvePnP(object_points.T,corner_points.T, K, distCoeffs=None)

            RT=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
            RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
            save_RT.append(RT)
    
        else:
            print("no figure find")
        
        cv2.imshow('Chessboard Corners', image)
        cv2.waitKey(2)
    # save_RT = np.array(save_RT)
    #     corner_points=np.zeros((2,corners.shape[0]),dtype=np.float64)
    #     for i in range(corners.shape[0]):
    #         corner_points[:,i]=corners[i,0,:]
        
    #     object_points=np.zeros((3,chess_board_x_num*chess_board_y_num),dtype=np.float64)
    #     flag=0
    #     for i in range(chess_board_y_num):
    #         for j in range(chess_board_x_num):
    #             object_points[:2,flag]=np.array([(11-j-1)*chess_board_len,(8-i-1)*chess_board_len])
    #             flag+=1
        
    #     retval,rvec,tvec  = cv2.solvePnP(object_points.T,corner_points.T, K, distCoeffs=None)

    #     RT=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
    #     RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    #     save_RT.append(RT)
    # save_RT = np.array(save_RT)
    cv2.destroyAllWindows()
    
    corner_points=np.zeros((2,corners.shape[0]),dtype=np.float64)
    for i in range(corners.shape[0]):
        corner_points[:,i]=corners[i,0,:]
    
    object_points=np.zeros((3,chess_board_x_num*chess_board_y_num),dtype=np.float64)
    flag=0
    for i in range(chess_board_y_num):
        for j in range(chess_board_x_num):
            object_points[:2,flag]=np.array([(11-j-1)*chess_board_len,(8-i-1)*chess_board_len])
            flag+=1
    
    retval,rvec,tvec  = cv2.solvePnP(object_points.T,corner_points.T, K, distCoeffs=None)

    RT=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))

    return RT


#用于根据欧拉角计算旋转矩阵,输入的欧拉角为弧度制
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx  #@表示矩阵乘法
    return R


#用于根据位姿计算变换矩阵
def pose_robot(x, y, z, Tx, Ty, Tz):   #旋转角度：x,y,z(rad);平移分量：Tx,Ty,Tz(m)
    thetaX = x
    thetaY = y
    thetaZ = z
    R = myRPY2R_robot(thetaX,thetaY,thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    return RT1


def chessboard_to_cam(external_image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len):
        #计算chessboard to cam 变换矩阵
        R_all_chess_to_cam_1=[]
        T_all_chess_to_cam_1=[]
        
        RT=get_RT_from_chessboard(external_image_path, chess_board_x_num, chess_board_y_num, chess_board_len, K)
        # print(RT[:3,:3])
        # print(RT[:3, 3])
        R_all_chess_to_cam_1.append(RT[:3,:3])
        T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3,1)))
        return R_all_chess_to_cam_1, T_all_chess_to_cam_1


def end_to_base(external_image_path):
        R_all_end_to_base_1=[]
        T_all_end_to_base_1=[]
        pose = []
        angle = []

        images = read_images_from_folder(external_image_path)

        for image in images:
            #角度为弧度，坐标为米
            rotation, phi, theta, psi, x, y, z = get_EE_pos()
            pose.append(np.array([x, y, z])) 
            angle.append(np.array([phi, theta, psi]))
        pose = np.array(pose, dtype='float64')
        angle = np.array(angle, dtype='float64')
        # print(pose.shape)
        # print(angle.shape)
        RT_=pose_robot(angle[:, 0],angle[:, 1],angle[:, 2],pose[:, 0],pose[:, 1],pose[:, 2])
        RT = np.linalg.inv(RT_)  # 这里加一步求逆，求的其实是base2end
        R_all_end_to_base_1.append(RT[:3, :3])
        # T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))
        T_all_end_to_base_1.append(RT[:3, 3])

        # pose = np.array(pose, dtype=np.float64)
        # angle = np.array(angle, dtype=np.float64)
        return R_all_end_to_base_1, T_all_end_to_base_1


def calculate_external(external_image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len):
        R_all_chess_to_cam_1, T_all_chess_to_cam_1 = chessboard_to_cam(external_image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)
        R_all_end_to_base_1, T_all_end_to_base_1 = end_to_base()
        # print(type(R_all_chess_to_cam_1))
        # print(type(R_all_end_to_base_1))
        R_all_chess_to_cam_1 = np.array(R_all_chess_to_cam_1)
        T_all_chess_to_cam_1 = np.array(T_all_chess_to_cam_1)
        R_all_end_to_base_1 = np.array(R_all_end_to_base_1)
        T_all_end_to_base_1 = np.array(T_all_end_to_base_1)

        R_all_chess_to_cam_1 = np.squeeze(R_all_chess_to_cam_1, 0)
        T_all_chess_to_cam_1 = np.squeeze(T_all_chess_to_cam_1, 0)
        R_all_end_to_base_1 = np.squeeze(R_all_end_to_base_1, 0)
        T_all_end_to_base_1 = np.squeeze(T_all_end_to_base_1, 0)

        R,T=cv2.calibrateHandEye(R_all_end_to_base_1,T_all_end_to_base_1,R_all_chess_to_cam_1,T_all_chess_to_cam_1)#手眼标定
        RT=np.column_stack((R,T))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))#即为cam to end变换矩阵
        print('相机相对于末端的变换矩阵为：')
        print(RT)
        return RT
    

def save_image(): 
    # 配置
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
 
    i = 0
    profile = pipe.start(cfg)
    save_phi = []
    save_theta = []
    save_psi = []
    save_x = []
    save_y = []
    save_z = []

    save_image_path = '/home/user/Downloads/furniture-bench-main/save_chessboard/end_external'
    # save_image_path = '/home/user/Downloads/furniture-bench-main/save_apriltag'
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
 
    while True:
        
        # 获取图片帧
        frameset = pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        color_img = np.asanyarray(color_frame.get_data())
        #交换颜色通道
        t = color_img[:,:, 2].copy()
        color_img[:,:, 2] = color_img[:,:, 0]
        color_img[:,:, 0] = t
    
        # 更改通道的顺序为RGB
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_img)
        k = cv2.waitKey(1)
        
        # Esc退出，
        if i == 15:
            cv2.destroyAllWindows()
            break
        # 输入空格保存图片
        elif k == ord(' '):
            i = i + 1
            cv2.imwrite(os.path.join(save_image_path, str(i) + '.jpg'), color_img)
            print("Frames{} Captured".format(i))
            rotation, phi, theta, psi, x, y, z = get_EE_pos()
            print(rotation, phi, theta, psi, x, y, z)
            
            save_phi.append(phi)
            save_theta.append(theta)
            save_psi.append(theta)
            save_x.append(theta)
            save_y.append(theta)
            save_z.append(theta)
            
    print("save_phi: ", save_phi)
    save_phi = np.array(save_phi)
    save_theta = np.array(save_theta)
    save_phi = [x for x in save_phi]
    save_theta = [y for y in save_theta]
    save_psi = [x for x in save_psi]
    save_x = [x for x in save_x]
    save_y = [x for x in save_y]
    save_z = [x for x in save_z]
    data = pd.DataFrame({'phi':save_phi, 'theta':save_theta,'psi':save_psi,'x':save_x,'y':save_y,'z':save_z})
    with open(save_image_path + 'robot_position.csv', 'a+', encoding='utf-8') as f:
        data.to_csv(f, index=False)
    # save_phi.reshape(-1, 1) 
    # save_theta.reshape(-1, 1)
    # # phi = np.array([phi])
    # # theta = np.array([theta])
    # # psi = np.array([psi])
    # # x = np.array([x])
    # # y = np.array([y])
    # # z = np.array([z])
    # # save_phi = pd.DataFrame(save_phi, columns=['phi'])
    # # save_theta = pd.DataFrame(save_theta, columns=['theta'])
    
    
    # with open(save_image_path + 'robot_position.csv', 'a+', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['phi', 'theta'])
    
    #     for i in range(len(save_phi)):
            
    #         writer.writerows([save_phi[i], save_theta[i]])
    #     # np.savetxt(save_image_path + 'robot_position.txt', 
    #     #         rotation, delimiter=',', encoding='utf-8')
    #     # np.savetxt(save_image_path + 'robot_position.txt', 
    #     #         phi, encoding='utf-8')
    #     # np.savetxt(save_image_path + 'robot_position.txt', 
    #     #         theta, encoding='utf-8')
    #     # np.savetxt(save_image_path + 'robot_position.txt', 
    #     #         psi, encoding='utf-8')
    #     # np.savetxt(save_image_path + 'robot_position.txt', 
    #     #         x, encoding='utf-8')
    #     # np.savetxt(save_image_path + 'robot_position.txt', 
    #     #         y, encoding='utf-8')
    #     # np.savetxt(save_image_path + 'robot_position.txt', 
    #     #         z, encoding='utf-8')
    #     # save_phi.to_csv(f, index=False)
    #     # save_theta.to_csv(f, index=False)
    #     # np.savetxt(f, 
    #     #         rotation, delimiter=',', encoding='utf-8')
    #     # np.savetxt(f, 
    #     #         phi, encoding='utf-8')
    #     # np.savetxt(f, 
    #     #         theta, encoding='utf-8')
    #     # np.savetxt(f, 
    #     #         psi, encoding='utf-8')
    #     # np.savetxt(f, 
    #     #         x, encoding='utf-8')
    #     # np.savetxt(f, 
    #     #         y, encoding='utf-8')
    #     # np.savetxt(f,  
    #     #         z, encoding='utf-8')

                
    pipe.stop()


    # 遍历标定板的图片
def read_images_from_folder(image_path):
    images = []
    i = 0
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(image_path, filename)
            # print(path)
            try:
                # image = Image.open(path)
                # images.append(image)
                images.append(path)
            except IOError:
                print("Cannot open image: ", filename)
    return images

# def calculate_intrinsic(image_path):
    
#     # 设置标定板的size和square
#     '''
#     如果是8x7,size填的就是7x6。square指的是标定板上每个格子的宽度,单位是m。
#     '''
#     board_size = (11, 8)
#     square_width = .02  # 假设每个格子的宽度为2厘米

#     # 创建并计算棋盘格每一个格的3维坐标
#     obj_points = []
#     objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
#     # objp[:, :2] = np.mgrid[:board_size[1], :board_size[2]].T.reshape(-1, 2) * square_width
#     objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_width
#     # 初始化数组用于存储世界坐标点和图像坐标点(即3D和2D图像点)
#     obj_points_list = []
#     img_points_list = []

#     # 实例化一个Pipeline对象，从相机中读取帧并处理数据流
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
#     config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

#     profile = pipeline.start(config)

#     images = read_images_from_folder(image_path)
#     # print(len(images))

#     # 遍历所有标定图像
#     for image in images:


#         # 读取图像并将其转换为灰度图
        
#         image = cv2.imread(image)
#         # print(image)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # 查找棋盘格角点
#         ret, corners = cv2.findChessboardCorners(gray, board_size, None)

#         # 如果找到棋盘格角点则存储对应的3D和2D坐标
#         if ret:
#             obj_points_list.append(objp)
#             img_points_list.append(corners)

#             # 在图像上绘制棋盘格角点
#             cv2.drawChessboardCorners(image, board_size, corners, ret)
#             cv2.imshow('Chessboard Corners', image)
#             cv2.waitKey(2)

#     cv2.destroyAllWindows()
#     # 进行相机内参标定
#     ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, img_points_list, gray.shape[::-1], None, None)

#     # 打印相机内参和畸变系数
#     # 如果需要保存多个相机的内参矩阵，则需要注意记录下相机编号
#     print("Camera Matrix:\n", camera_matrix)
#     print("\nDistortion Coefficients:\n", dist_coeffs)
#     np.savetxt(image_path + 'camera_matrix.txt', camera_matrix)
#     np.savetxt(image_path + 'dist_coeffs.txt', dist_coeffs)

#     pipeline.stop()

class calculate_external():
    def __init__(self):
        self.num = 15  #用于标定的图片数，需要根基实际保存的图片数进行修改

        #相机内参，需要根据实际内参大小进行修改
        self.fx = 4.769347547203772137e+02
        self.fy = 4.776687745938567673e+02
        self.cx = 3.088977531883862753e+02
        self.cy = 1.795221462104951229e+02
        
        #用于根据欧拉角计算旋转矩阵,输入的欧拉角为弧度制
    def myRPY2R_robot(self, x, y, z):
        Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        R = Rz@Ry@Rx  #@表示矩阵乘法
        return R         #[0,0,1]],dtype=np.float64)
 
        #棋盘格参数，需要根据实际使用的棋盘格格数来设定
        self.chess_board_x_num=11
        self.chess_board_y_num=8
        self.chess_board_len=20.0  #单位棋盘格长度,mm,这里要修改
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) #用于查找棋盘格角点
        # 需要替换为实际的标定板保存路径，并在保存时需要注意将图片对应的机械臂末端姿态进行记录
        #self.external_image_path = '/home/user/Downloads/furniture-bench-main/save_apriltag'
        self.external_image_path = '/home/user/Downloads/furniture-bench-main/save_chessboard/test_camera_external'
        # 需要替换为实际的标定板保存路径，并在保存时需要注意将图片对应的机械臂末端姿态进行记录
        self.image_path = '/home/user/Downloads/furniture-bench-main/save_apriltag/1.jpg'

        # # 实例化一个Pipeline对象，从相机中读取帧并处理数据流
        # self.pipeline = rs.pipeline()
        # self.config = rs.config()
        # self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

        # self.profile = self.pipeline.start(self.config)
 
    """
    根据欧拉角计算旋转矩阵
    """
    #用于根据欧拉角计算旋转矩阵,输入的欧拉角为弧度制
    def myRPY2R_robot(self, x, y, z):
        Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        R = Rz@Ry@Rx  #@表示矩阵乘法
        return R
    
    """
    根据机械臂的平移和旋转计算齐次矩阵
    """
    #用于根据位姿计算变换矩阵
    def pose_robot(self, x, y, z, Tx, Ty, Tz):   #旋转角度：x,y,z(rad);平移分量：Tx,Ty,Tz(m)
        thetaX = x
        thetaY = y
        thetaZ = z
        R = self.myRPY2R_robot(thetaX,thetaY,thetaZ)
        t = np.array([[Tx], [Ty], [Tz]])
        RT1 = np.column_stack([R, t])  # 列合并
        RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
        return RT1
 
 
    """
    根据棋盘格图像获取标定板相对于相机的位姿
    """
    #用来从棋盘格图片得到相机外参
    def get_RT_from_chessboard(self):
        images = read_images_from_folder(self.external_image_path)
        
        save_corners = []
        # 遍历所有标定图像
        for image in images:

            # 读取图像并将其转换为灰度图
            
            image = cv2.imread(image)
            # print(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (self.chess_board_x_num, self.chess_board_y_num), None)

            # 如果找到棋盘格角点则存储对应的3D和2D坐标
            if ret:
                # 精细查找角点
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(image, (self.chess_board_x_num, self.chess_board_y_num), corners2, ret)  #调换以后方向才对
                
                save_corners.append(corners)

            cv2.imshow('Chessboard Corners', image)
            cv2.waitKey(2)
        cv2.destroyAllWindows()

        # corner_points=np.zeros((2,corners.shape[0]),dtype=np.float64)
        
        corner_points=np.zeros((2,corners.shape[0]),dtype=np.float64)
        for i in range(corners.shape[0]):
            corner_points[:,i]=corners[i,0,:]
        
        object_points=np.zeros((3,self.chess_board_x_num*self.chess_board_y_num),dtype=np.float64)
        flag=0
        for i in range(self.chess_board_y_num):
            for j in range(self.chess_board_x_num):
                object_points[:2,flag]=np.array([(11-j-1)*self.chess_board_len,(8-i-1)*self.chess_board_len])
                flag+=1
        
        retval,rvec,tvec  = cv2.solvePnP(object_points.T,corner_points.T, self.K, distCoeffs=None)
    
        RT=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    
        return RT
 
 
 
    """
    标定板->相机
    """
    def chessboard_to_cam(self):
        #计算chessboard to cam 变换矩阵
        R_all_chess_to_cam_1=[]
        T_all_chess_to_cam_1=[]
        for i in range(self.num):
            
            RT=self.get_RT_from_chessboard(self.image_path, self.chess_board_x_num, self.chess_board_y_num, self.K, self.chess_board_len)
        
            R_all_chess_to_cam_1.append(RT[:3,:3])
            T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3,1)))
        return R_all_chess_to_cam_1, T_all_chess_to_cam_1
 
    """
    末端->基座
    """
    def end_to_base(self):
        R_all_end_to_base_1=[]
        T_all_end_to_base_1=[]
        pose = []
        angle = []
        for i in range(self.num):   
            #角度为弧度，坐标为米
            rotation, phi, theta, psi, x, y, z = get_EE_pos()
            pose.append(np.array([x, y, z])) 
            angle.append(np.array([phi, theta, psi]))
            RT_=self.pose_robot(angle[i,0],angle[i,1],angle[i,2],pose[i,0],pose[i,1],pose[i,2])
            RT = np.linalg.inv(RT_)  # 这里加一步求逆，求的其实是base2end
            R_all_end_to_base_1.append(RT[:3, :3])
            # T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))
            T_all_end_to_base_1.append(RT[:3, 3])

        # pose = np.array(pose, dtype=np.float64)
        # angle = np.array(angle, dtype=np.float64)
        return R_all_end_to_base_1, T_all_end_to_base_1
        
#         # #位姿参数，修改
#         # pose=np.array([[-622.4452510680815,524.9752698923884,156.06834188650686],
#         #     [-627.8618566771845,520.3385452748523,155.95649892911888],
#         #     [-627.8603117576374,520.3395366048145,155.96015182265455],
#         #     [-608.5479357433793,542.7989266050945,155.9582535151922],
#         #     [-608.73900609math.92511,543.0210484916622,151.91097181613594],
#         #     [-602.8760786176158,549.5229480943764,151.91097181613594],
#         #     [-603.552411795536,550.3470321566062,133.70817720248263],
#         #     [-611.5465863717345,541.1627275715787,134.34846121499896],
#         #     [-611.5443671647014,541.1628776214664,134.34718510333377],
#         #     [-602.7423615676076,550.5760656441264,143.8503303749094]])
#         # ang=np.array([[-3.0398882702599774,1.044591855511297,-0.7693595597257228],
#         #     [-3.066247439671175,1.0454779707979422,-0.7692025753376092],
#         #     [-3.122619548308192,1.047022325344027,-0.8343332090498904],
#         #     [-3.122607921681094,1.0470154640492697,-0.8706462420205126],
#         #     [-3.122629578093413,1.0420052892962037,-0.8706604603523166],
#         #     [-3.122629578093413,1.0420052892962037,-0.8813929731338211],
#         #     [-3.083008314702459,1.018524644841849,-0.8348392811215427],
#         #     [-3.1262532894037527,1.0331396840313745,-0.8336434111432051],
#         #     [3.119411368295909,1.033064358179091,-0.8773344026002032],
#         #     [-3.11584155786669,1.0667034468346508,-0.8769577446455527]])
    
 
#     """
#     外参标定
#     """
    def calculate_external(self):
        R_all_chess_to_cam_1, T_all_chess_to_cam_1 = self.chessboard_to_cam()
        R_all_end_to_base_1, T_all_end_to_base_1 = self. end_to_base()
        R,T=cv2.calibrateHandEye(R_all_end_to_base_1,T_all_end_to_base_1,R_all_chess_to_cam_1,T_all_chess_to_cam_1)#手眼标定
        RT=np.column_stack((R,T))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))#即为cam to end变换矩阵
        print('相机相对于末端的变换矩阵为：')
        print(RT)
        return RT
    
 
# # """
# # 结果验证
# # """
# # #结果验证，原则上来说，每次结果相差较小
# # def test(self):
# #     for i in range(self.num):
    
# #         RT_end_to_base=np.column_stack((R_all_end_to_base_1[i],T_all_end_to_base_1[i]))
# #         RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    
# #         RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1[i],T_all_chess_to_cam_1[i]))
# #         RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    
# #         RT_cam_to_end=np.column_stack((R,T))
# #         RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
    
# #         RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam  #即为固定的棋盘格相对于机器人基坐标系位姿
# #         RT_chess_to_base=np.linalg.inv(RT_chess_to_base)
# #         print('第',i,'次')
# #         print(RT_chess_to_base[:3,:])
# #         print('')
 

if __name__ == "__main__":
    # cv2.destroyAllWindows()
    # save_image()
    # image_path = '/home/user/Downloads/furniture-bench-main/save_chessboard/Camera_5'
    # # images = read_images_from_folder(image_path)
    # calculate_intrinsic(image_path)

  
    external_image_path = '/home/user/Downloads/furniture-bench-main/save_chessboard/Camera_4'
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

    RT = get_RT_from_chessboard(external_image_path, chess_board_x_num, chess_board_y_num, chess_board_len, K)
    print(RT)
    # # print(RT[:3,:3])
    # # print(RT[:3, 3])

    # R_all_chess_to_cam_1, T_all_chess_to_cam_1 = chessboard_to_cam(external_image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)
    # print(R_all_chess_to_cam_1, T_all_chess_to_cam_1)

    # R_all_end_to_base_1, T_all_end_to_base_1 = end_to_base()
    # # print(R_all_end_to_base_1, T_all_end_to_base_1)

    # rt = calculate_external(external_image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)
    # # print(rt)


