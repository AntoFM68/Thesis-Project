#!/bin/bash
if [ $1 == "" ]; then
	echo "Error: you must supply the image name"
	exit 0
fi

IMAGE=$1

xhost +; docker run  --gpus all \
		--rm \
		-it \
		--privileged \
		--ipc=host \
		--network=host \
		--device=/dev/dri:/dev/dri \
		-v /dev:/dev \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $PWD/AWS_Deepracer:/root/catkin_ws/src \
		-v $PWD/RL_Lab:/root/rl_lab \
		-v $PWD/Mujoco_car_model:/root/.mujoco/mujoco210/model \
        -v $PWD/joystick.config.yaml:/opt/ros/foxy/share/teleop_twist_joy/config/xbox.config.yaml \
		-e DISPLAY=$DISPLAY \
		-v $HOME/.Xauthority:/home/$(id -un)/.Xauthority \
		-e XAUTHORITY=/home/$(id -un)/.Xauthority \
		$IMAGE /bin/bash
