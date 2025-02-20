# Start with an Ubuntu 22.04 ros-humble image
FROM osrf/ros:humble-desktop-full

# Set shell to bash and prevent dpkg frontend dialog
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Source the ROS 2 setup.bash in bashrc for interactive shells
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-gazebo-ros2-control \
    ros-humble-controller-manager \
    ros-humble-joint-state-publisher-gui \
    ros-humble-imu-tools \
    ros-humble-tf2-tools \
    ros-humble-moveit \
    nano && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyQt5 using pip
RUN pip3 install --no-cache-dir PyQt5
RUN pip3 install --no-cache-dir transforms3d 
RUN pip3 install --no-cache-dir pyserial
RUN pip3 install --no-cache-dir wlkatapython

# Entry point to ensure container remains running
CMD ["bash"]