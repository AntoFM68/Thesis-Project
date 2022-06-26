# RL_lab Docker image
1. cd /Docker
2. docker run --rm -it --gpus=all -e DISPLAY=host.docker.internal:0.0 -e LIBGL_ALWAYS_INDIRECT=0 -v ${pwd}:/root/rl_lab git.quavlive.com:5052/wb-phd/code rl_lab bash

# RL & ROS2 Docker image 
1. cd Docker
2. docker run --rm -it -P --gpus=all -e DISPLAY=host.docker.internal:0.0 -e LIBGL_ALWAYS_INDIRECT=0 -v ${pwd}/AWS_Deepracer:/root/catkin_ws/src rl_ros2 bash 

# RL_ROS2 run dr_ctrl_py
1. colcon build --packages-select deepracer_interfaces_pkg
2. colcon build --packages-select dr_ctrl_py
3. . install/setup.bash **in catkin_ws**

**single mapping**
ros2 run dr_ctrl_py mapper 0.0 0.0 

**joystick teleop**   
1. ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox'
2. ros2 run dr_ctrl_py joy

**battery level**
ros2 run dr_ctrl_py battery

**camera image and video**
1. ros2 run dr_ctrl_py image
2. ros2 run dr_ctrl_py video

# Mujoco simulation
1. cd ..
2. cd .mujoco/mujoco210/
3. ./bin/simulate /root/.mujoco/mujoco210/model/car/AWS_Deepracer/deepracer.xml

# Open active image in multiple terminal
1. docker ps
2. docker exec -it container_id bash

# Remove <none> images
1. docker image ls -f 'dangling=true'
2. docker image rm $(docker image ls -f 'dangling=true' -q)
