import sys
sys.path.append('/home/user/franka_ws/devel/lib/python3/dist-packages')
from geometry_msgs.msg import PoseStamped, WrenchStamped
from franka_msgs.msg import FrankaState
import rospy
import numpy as np
from sensor_msgs.msg import JointState
import math


def euler_to_quaternion(roll, pitch, yaw):
    # Compute half-angles
    roll_half = roll / 2
    pitch_half = pitch / 2
    yaw_half = yaw / 2

    # Compute trigonometric functions of half-angles
    cr = np.cos(roll_half)
    sr = np.sin(roll_half)
    cp = np.cos(pitch_half)
    sp = np.sin(pitch_half)
    cy = np.cos(yaw_half)
    sy = np.sin(yaw_half)

    # Compute quaternion components
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz       


rospy.init_node('mynode')


def get_dq():
    state = rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState)
    dq=state.dq
    return dq


def get_EE_pos():  # get end effector pose
    state = rospy.wait_for_message('/franka_state_controller/franka_states', FrankaState)
    rotation = state.O_T_EE # rotation matrix
    
    # print(state)
    x = rotation[12]
    y = rotation[13]
    z = rotation[14]

    print("x:{}".format(x))
    print("y:{}".format(y))
    print("z:{}".format(z))

    R = np.array(rotation)

    R = R.reshape((4, 4))
    R = np.delete(R, -1, axis=0)
    R = np.delete(R, -1, axis=1)

    phi, theta, psi = rotation_matrix_to_euler(R)
    print("Roll (phi): {:.4f} radians".format(phi))
    print("Pitch (theta): {:.4f} radians".format(theta))
    print("Yaw (psi): {:.4f} radians".format(psi))
    eef_pose = np.array([[x],[y],[z]]).reshape((-1,1))
    return rotation, phi, theta, psi, x, y, z, eef_pose  # rotation matrix, roll, pitch, yaw(radians), x, y, z(meter)
    

def get_joint_angle():  
    
    joint = rospy.wait_for_message('/franka_state_controller/joint_states', JointState)
    # print(joint.position)
    return joint.position, joint.velocity, joint.effort

def calculate_robot_joint():
    joint_angles, joint_velocities, joint_torques = get_joint_angle()
    joint_angles = np.array([joint_angles])
    joint_velocities = np.array([joint_velocities])
    joint_torques = np.array([joint_torques])
    return joint_angles, joint_velocities, joint_torques


def dh_matrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def joint_to_ee(action):
    L1 = 0.333
    L2 = 0.316
    L3 = 0.384
    L4 = 0.107
    L5 = 0.102
    L6 = 0.102
    L7 = 0.039
    # 假设有7个关节，使用DH参数表示
    dh_params = [
        (0, 0.5 * math.pi, 0, action[0]),
        (0, -0.5 * math.pi, L1, action[1]),
        (0, 0.5 * math.pi, 0, action[2] + 0.5 * math.pi),
        (L2, 0.5 * math.pi, 0, action[3]),
        (L3, -0.5 * math.pi, 0, action[4]),
        (L4, 0.5 * math.pi, 0, action[5]),
        (L5, 0, 0, action[6] + math.pi),
    ]
    # dh_params = [
    #     (theta1, d1, a1, alpha1),
    #     (theta2, d2, a2, alpha2),
    #     (theta3, d3, a3, alpha3),
    #     (theta4, d4, a4, alpha4),
    #     (theta5, d5, a5, alpha5),
    #     (theta6, d6, a6, alpha6),
    #     (theta7, d7, a7, alpha7),
    # ]

    # 计算末端执行器的变换矩阵
    T = np.eye(4)
    for params in dh_params:
        T = np.dot(T, dh_matrix(*params))

    T[2, 3] += L6 + L7
    # 末端执行器的位置
    end_effector_position = T[3, :3]

    # 末端执行器的姿态（四元数表示）
    end_effector_orientation = T[:3, :3]

    print("末端执行器的位置: ", end_effector_position)
    print("末端执行器的姿态: ", end_effector_orientation)

def rotation_matrix_to_euler(R):

    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 numpy array")

    # # Check if the matrix is a proper rotation matrix
    # if not np.allclose(np.dot(R, R.T), np.eye(3)) or not np.isclose(np.linalg.det(R), 1):
    #     raise ValueError("Input matrix is not a valid rotation matrix")

    # Extract elements from the rotation matrix
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]

    # Compute Euler angles
    if np.isclose(r31, 1) or np.isclose(r31, -1):  # Gimbal lock condition
        theta = np.pi / 2 if r31 > 0 else -np.pi / 2
        phi = 0
        psi = np.arctan2(-r12, r22)
    else:
        theta = np.arctan2(-r31, np.sqrt(r32 ** 2 + r33 ** 2))
        phi = np.arctan2(r32, r33)
        psi = np.arctan2(r21, r11)

    return phi, theta, psi


if __name__ == "__main__":
    rospy.init_node('mynode')
    cmd_pub = rospy.Publisher('/cartesian_impedance_controller/desired_pose', PoseStamped, queue_size=10)

    desired_pose = [0.4,0.0,0.1,3.14,0,0]  # x, y, z(meter), roll, pitch, yaw(radians)

    pos_cmd = PoseStamped()
    pos_cmd.pose.position.x, pos_cmd.pose.position.y, pos_cmd.pose.position.z = desired_pose[0:3]
    pos_cmd.pose.orientation.w, pos_cmd.pose.orientation.x, pos_cmd.pose.orientation.y, pos_cmd.pose.orientation.z = euler_to_quaternion(
                                                                        desired_pose[3], desired_pose[4], desired_pose[5])
                                                                        
    cmd_pub.publish(pos_cmd)

    rotation, phi, theta, psi, x, y, z = get_EE_pos()
    # print(rotation, phi, theta, psi, x, y, z)
    position = get_joint_angle()
    print(position)