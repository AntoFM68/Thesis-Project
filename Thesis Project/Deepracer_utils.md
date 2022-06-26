ssh deepracer@10.73.0.113     **c3lab**
ssh deepracer@192.168.16.103  **officine**
password: deepracer

sudo ufw disable
sudo su
cd
source setup_env.sh
source /opt/intel/openvino_2021/bin/setupvars.sh
ros2 launch deepracer_launcher deepracer_launcher.py

# Control launch #
source ~/deepracer_ws/aws-deepracer-ctrl-pkg/install/setup.bash
ros2 launch ctrl_pkg ctrl_pkg_launch.py

# LED (valori: 10.000.000) #
ros2 service call /servo_pkg/set_led_state deepracer_interfaces_pkg/srv/SetLedCtrlSrv '{red: 0, green: 0, blue: 0}'

# Enable GPIO (enable: 0, disable: 1) #
ros2 service call /servo_pkg/servo_gpio deepracer_interfaces_pkg/srv/ServoGPIOSrv '{enable: 0}'

# Control angle and throttle #
ros2 topic pub /ctrl_pkg/servo_msg deepracer_interfaces_pkg/msg/ServoCtrlMsg '{angle: 0, throttle: 0}'
ros2 topic pub /ctrl_pkg/raw_pwm deepracer_interfaces_pkg/msg/ServoCtrlMsg '{angle: 1625000, throttle: 1600000}'

# Calibrate angle and throttle #
ros2 service call /servo_pkg/get_calibration deepracer_interfaces_pkg/srv/GetCalibrationSrv "{cal_type: 0}" **0: angle, 1: throttle**
ros2 service call /servo_pkg/set_calibration deepracer_interfaces_pkg/srv/SetCalibrationSrv "{cal_type: 0, max: 1800000, mid: 1625000, min: 1400000, polarity: 1}"
ros2 service call /servo_pkg/set_calibration deepracer_interfaces_pkg/srv/SetCalibrationSrv "{cal_type: 1, max: 1680000, mid: 1550000, min: 1450000, polarity: 1}"

# Battery #
ros2 service call /i2c_pkg/battery_level deepracer_interfaces_pkg/srv/BatteryLevelSrv

# Camera #
source ~/deepracer_ws/aws-deepracer-camera-pkg/install/setup.bash
ros2 launch camera_pkg camera_pkg_launch.py
ros2 service call /camera_pkg/media_state deepracer_interfaces_pkg/srv/VideoStateSrv "{activate_video: 1}"

### setup_env.sh ###
source /opt/ros/foxy/setup.bash
source ~/deepracer_ws/aws-deepracer-launcher/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp



####################

sudo su
ufw disable
source /opt/ros/foxy/setup.bash
source ~/deepracer_ws/aws-deepracer-servo-pkg/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
unset ROS_LOCALHOST_ONLY
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
ros2 launch servo_pkg servo_pkg_launch.py

ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 12}"

MOTORI
ros2 service call /ctrl_pkg/enable_state deepracer_interfaces_pkg/srv/EnableStateSrv "{is_active: 1}"
ros2 topic pub /ctrl_pkg/ctrl_pkg deepracer_interfaces_pkg/msg/ServoCtrlMsg "{angle: 0.8, throttle: +0.8}"
