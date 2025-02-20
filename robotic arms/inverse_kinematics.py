import numpy as np
from math import atan2, sqrt, acos, pi, cos, sin

def degrees_to_radians(degrees):
    """Convert degrees to radians."""
    return [angle * pi / 180 for angle in degrees]

def rpy_to_rotation_matrix(roll, pitch, yaw):
    """Convert Roll-Pitch-Yaw angles to a 3x3 rotation matrix."""
    R_x = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])
    R_y = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])
    R_z = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

def inverse_kinematics(target_position, target_rpy, joint_translations, wrist_offset_x):
    """
    Compute the inverse kinematics for a 6-DOF robot arm with a spherical wrist.
    :param target_position: [x, y, z] - Desired end-effector position in mm.
    :param target_rpy: [roll, pitch, yaw] - Desired end-effector orientation in RPY angles (radians).
    :param joint_translations: List of translations for each joint in mm.
    :param wrist_offset_x: Offset from the wrist center to the end-effector in mm.
    :return: List of joint angles [theta1, theta2, theta3, theta4, theta5, theta6] in radians.
    """
    # Unpack target position and orientation
    x, y, z = target_position
    roll, pitch, yaw = target_rpy

    # Step 1: Compute the wrist center position
    # The wrist center is offset from the end-effector by wrist_offset_x along the x-axis of the end-effector frame
    R = rpy_to_rotation_matrix(roll, pitch, yaw)
    wrist_center = np.array([x, y, z]) - wrist_offset_x * R[:, 0]

    # Step 2: Solve for the first three joints (arm)
    xc, yc, zc = wrist_center

    # Joint 1: Rotation around the z-axis
    theta1 = atan2(yc, xc)

    # Joint 2 and 3: Solve using geometric approach
    # Extract joint translations
    a1z = joint_translations[0][2]
    a2x, a2z = joint_translations[1][0], joint_translations[1][2]
    a3z = joint_translations[2][2]
    a4x, a4z = joint_translations[3][0], joint_translations[3][2]
    a5x = joint_translations[4][0]

    # Calculate intermediate values
    WP_xy = sqrt(xc**2 + yc**2)
    l = WP_xy - a2x  # Assuming the offset is subtracted
    h = zc - a1z - a2z
    rho = sqrt(h**2 + l**2)
    b4x = sqrt(a4z**2 + (a4x + a5x)**2)

    # Calculate angles
    alpha = atan2(h, l)
    cos_beta = (rho**2 + a3z**2 - b4x**2) / (2 * rho * a3z)
    beta = acos(cos_beta)
    theta2 = pi/2 - alpha - beta  # Assuming the negative sign for beta

    cos_gamma = (a3z**2 + b4x**2 - rho**2) / (2 * a3z * b4x)
    gamma = acos(cos_gamma)
    delta = atan2(a4x + a5x, a4z)
    theta3 = pi - gamma - delta

    # Step 3: Solve for the last three joints (wrist)
    # Compute the rotation matrix for the arm (R_arm)
    c1, s1 = cos(theta1), sin(theta1)
    c23, s23 = cos(theta2 + theta3), sin(theta2 + theta3)
    R_arm = np.array([
        [c1 * c23, -s1, c1 * s23],
        [s1 * c23, c1, s1 * s23],
        [-s23, 0, c23]
    ])

    # Compute the wrist orientation (R_wrist)
    R_wrist = R_arm.T @ R

    # Extract wrist joint angles (J4, J5, J6) from R_wrist
    # Using the provided formulas
    theta4 = atan2(R_wrist[1, 0], -R_wrist[2, 0])  # J4 = atan2(R21, -R31)
    theta5 = atan2(sqrt(1 - R_wrist[0, 0]**2), R_wrist[0, 0])  # J5 = atan2(±√(1 - R11^2), R11)
    theta6 = atan2(R_wrist[0, 1], R_wrist[0, 2])  # J6 = atan2(R12, R13)

    return [theta1, theta2, theta3, theta4, theta5, theta6]

# Example usage
target_position = np.array([-500.0, 1000.0, 2000.0])  # Desired position in mm
target_orientation_rpy = degrees_to_radians([50.0, 50.0, 50.0])  # Desired orientation in RPY (radians)
wrist_offset_x = 244.0  # Assumed end-effector length from wrist center in mm
joint_translations = [
    [0.0, 0.0, 650.0],   # Translation for joint 1
    [400.0, 0.0, 680.0], # Translation for joint 2 (x and z offset)
    [0.0, 0.0, 1100.0],  # Translation for joint 3
    [766.0, 0.0, 230.0], # Translation for joint 4
    [345.0, 0.0, 0.0],   # Translation for joint 5
    [244.0, 0.0, 0.0]    # Translation for joint 6
]

joint_angles = inverse_kinematics(target_position, target_orientation_rpy, joint_translations, wrist_offset_x)
# print("Joint Angles (radians):", joint_angles)
print("Joint Angles (degrees):", [angle * 180 / pi for angle in joint_angles])