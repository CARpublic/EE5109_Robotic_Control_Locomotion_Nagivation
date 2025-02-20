import numpy as np

def rotation_matrix_x(theta):
    """Rotation matrix for rotation around the x-axis."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    """Rotation matrix for rotation around the y-axis."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    """Rotation matrix for rotation around the z-axis."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def translate(t):
    """Translation transformation matrix."""
    T = np.eye(4)
    T[:3, 3] = t
    return T

def transform(rotation, translation):
    """Homogeneous transformation matrix from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def rotation_matrix_to_rpy(R):
    """Convert a rotation matrix to roll, pitch, and yaw (RPY) angles."""
    # Assuming the rotation matrix R is for the ZYX rotation order (Yaw-Pitch-Roll)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:  # Non-singular case
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # Singular case
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def forward_kinematics(thetas, translations):
    """
    Compute forward kinematics for a 6-DOF robotic arm.
    
    Parameters:
        thetas (list of float): List of joint angles [theta1, theta2, ..., theta6].
        translations (list of list of float): Translation vectors [[x1, y1, z1], ..., [x6, y6, z6]].

    Returns:
        tuple: Position and orientation (RPY angles) of the TCP.
    """ 
    # Define each joint's transformation
    T1 = transform(rotation_matrix_z(thetas[0]), translations[0])
    T2 = transform(rotation_matrix_y(thetas[1]), translations[1])
    T3 = transform(rotation_matrix_y(thetas[2]), translations[2])
    T4 = transform(rotation_matrix_x(thetas[3]), translations[3])
    T5 = transform(rotation_matrix_y(thetas[4]), translations[4])
    T6 = transform(rotation_matrix_x(thetas[5]), translations[5])

    # Compute the overall transformation from the base to TCP
    T = np.eye(4)
  
    for Ti in [T1, T2, T3, T4, T5, T6]:
        T = np.dot(T, Ti)

    # Extract position and orientation in RPY
    position = T[:3, 3]
    RPY = np.degrees(rotation_matrix_to_rpy(T[:3, :3]))

    return position, RPY

# Example joint angles (in radians)
joint_angles = np.radians([130, -60, 30, 60, -90, 60])

# Example translation vectors for each joint
translations = [
    [0.0, 0.0, 650.0],  # Translation for joint 1
    [400.0, 0.0, 680.0],  # Translation for joint 2
    [0.0, 0.0, 1100.0],  # Translation for joint 3
    [766.0, 0.0, 230.0],  # Translation for joint 4
    [345.0, 0.0, 0.0],  # Translation for joint 5
    [244.0, 0.0, 0.0]   # Translation for joint 6
]

# Compute the TCP position and orientation
tcp_position, tcp_rpy = forward_kinematics(joint_angles, translations)

print("TCP Position:", tcp_position)          # Output the TCP position
print("TCP Orientation (RPY):", tcp_rpy)      # Output the TCP orientation in RPY