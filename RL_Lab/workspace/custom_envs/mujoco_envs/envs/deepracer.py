from cmath import cos
import os
import numpy as np
from numpy.core.records import array
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from custom_modules.utils import get_theta_from_quat, get_smallest_angle
from mujoco_py import cymj, MjRenderContext
from dowel import logger
import cv2


# Gather Info required for the State
def gather_info_mujoco(robot, default, main_body):

    # Compute General Info
    pose = robot.get_body_xpos(main_body)
    pose[2] = get_theta_from_quat(robot.get_body_xquat(main_body))
    vels = [*robot.get_body_xvelp(main_body)[:2], robot.get_body_xvelr(main_body)[2]]
    #              [Xdot, Ydot]                         Thetadot
    
    # The indeces of the sensors strictly depends on the order of definition in the .xml file.
    # In this case: accelerometer, velocimiter... 
    accelerometer = robot.sensordata[0:3]
    velocimeter = robot.sensordata[3:6]

    V = np.sqrt(vels[0]**2 + vels[1]**2)
    Vangle = np.arctan2(vels[1], vels[0])

    dist = np.round(np.sqrt((default[0] - pose[0])**2 + (default[1] - pose[1])**2), 4)

    info = {
        "POSE": pose,
        "VELS": vels,
        "AVG_REAL_VEL_INFO": [V, Vangle],
        "SENS_ACC": accelerometer,
        "SENS_VEL": velocimeter,
        "DIST": dist,
    }
    return info

_TRAIN_MODE = True
_TEST_MODE = False


class Deepracer(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.main_body = "aws_deepracer"
        self.image_w = 160
        self.image_h = 120
        self.state_shape = (self.image_h, self.image_w*2, 3)   # Should be specified for every new environment.
        self.action_shape = 2  # Should be specified for every new environment.
        self.default_pose = [0., 0., 0.]
        self.dist = 0
        self.step_size = 0.01

        self.r_cam = np.zeros((self.image_h, self.image_w, 3), dtype=np.float32)
        self.l_cam = np.zeros((self.image_h, self.image_w, 3), dtype=np.float32)
        self.r_filter = np.zeros((self.image_h, self.image_w))
        self.l_filter = np.zeros((self.image_h, self.image_w))

        # Select a goal sampling strategy
        self.goal = self.goal_track()

        # self.configs = get_general_params()
        self.nstep = 0
        # self.max_steps = kwargs["N_TRAIN_STEPS"]
        self.reward_proportionals = {
            "DONE_REWARD": 30,
            "TRACK": 1,
            "LIMIT": 0.1,

        }
        assert type(self.reward_proportionals) == dict, " Bad Reward Proportionals: {}".format(type(self.reward_proportionals))

        self.info = None

        self.exec_mode = _TRAIN_MODE

        self.init_pose = self.default_pose
        self.pose = self.init_pose

        # Note: if model_path starts with "/", it will be considered as the full path.
        #       otherwise, it will be considered relative to "assets" folder (in the gym path installation).
        #       If not found, will raise an error.
        assests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        mujoco_env.MujocoEnv.__init__(self, model_path=os.path.join(assests_dir,"circuit.xml"), frame_skip=1)
        # cymj.set_pid_control(self.model, self.data) # FROM: https://github.com/openai/mujoco-py/blob/master/mujoco_py/tests/test_pid.py
        utils.EzPickle.__init__(self)

        self.observation_space = self._set_observation_space()
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Flag to understand when an episode has reached an end. (Used to call the evaluation of the episode)
        self._ep_fin = False

        # self.f_render = MjRenderContext(sim=self.sim, device_id=1, offscreen=True, opengl_backend='glfw')
        self.r_render = MjRenderContext(sim=self.sim, device_id=2, offscreen=True, opengl_backend='glfw')
        self.l_render = MjRenderContext(sim=self.sim, device_id=3, offscreen=True, opengl_backend='glfw')
        



    def _set_observation_space(self, ob=None):
        # For simplicity, bounds are set to be -/+ inf.
        low = np.full(self.state_shape, -float("inf"), dtype=np.float32)
        high = np.full(self.state_shape, float("inf"), dtype=np.float32)
        observation_space = spaces.Box(low, high, dtype=np.float32)
        return observation_space

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset_model(self):
        self._n_step = 0
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)  # MujocoEnv default func

        self.init_pose = self.sim.data.get_body_xpos(self.main_body)
        self.init_pose[2] = get_theta_from_quat(self.sim.data.get_body_xquat(self.main_body))
        self.pose = self.init_pose

        ob = self._get_obs()
        return ob

    def step(self, action):
        # Get Current State | Apply Action | # Get New State

        # print('Action:', action)

        self.do_simulation(action, self.frame_skip)
        self.nstep += 1

        obs = self._get_obs()
        reward, more_info = self._reward()

        # print('Reward:', reward)

        done = self._done_cond(reward)
        win = self.win()
        if win:
            reward += self.reward_proportionals["DONE_REWARD"]

        # Add some more info here
        more_info["outcome"] = "Success" if win else "Failure"
        more_info["DIST"] = self.dist
        return (obs, reward, done, more_info)
    
    def _get_obs(self):
        robot = self.sim.data
        self.info = gather_info_mujoco(robot, self.default_pose, main_body=self.main_body)
        self.dist = self.info["DIST"]

        # print(self.info["AVG_REAL_VEL_INFO"])

        ob = np.hstack((self.l_cam, self.r_cam,))
        assert ob.shape == self.state_shape, "Observation is {} but should be ({},) instead.".format(ob.shape, self.state_shape)
        return ob

    def _reward(self):
        r_mean = np.mean(self.r_filter[100:120][:], axis=0)
        r_track = np.mean(r_mean[80:100])
        r_limit= np.mean(r_mean[100:130])
        l_mean = np.mean(self.l_filter[100:120][:], axis=0)
        l_track = np.mean(l_mean[60:80])
        l_limit= np.mean(l_mean[30:60])
        # velocity reward
        if(self.info["AVG_REAL_VEL_INFO"][0] > 0.7):
            vel = 5
        else: 
            vel = 0
        cost = (vel - r_track - l_track) * self.reward_proportionals["TRACK"] - (r_limit + l_limit) * self.reward_proportionals["LIMIT"]
        more_info = {}
        return cost, more_info

    def goal_track(self):
        r_mean = np.mean(self.r_filter[100:120][:], axis=0)
        r_mean = np.mean(r_mean)
        l_mean = np.mean(self.l_filter[100:120][:], axis=0)
        l_mean = np.mean(l_mean)
        if (r_mean >= 110 and l_mean >= 110):
            print('Finish Line! Goal reached')
            return True

    def _done_cond(self, reward):
        if self.goal_track():
            return True
        elif reward < -160:
            print('Failure! Out of the track.')
            return True
    
    def win(self):
        return self.goal_track()

    def set_test_mode(self):
        self.exec_mode = _TEST_MODE        # TEST = False | TRAIN = True

    def set_train_mode(self):    # Redundant: Training mode is default
        self.exec_mode = _TRAIN_MODE



    def render_camera(self):
        # self.f_render.render(160, 120, camera_id=-1)
        # free_cam = self.f_render.read_pixels(160, 120, depth=False)
        # free_cam = free_cam[::-1, :, ::-1]

        self.r_render.render(160, 120, camera_id=2)
        self.r_cam = self.r_render.read_pixels(160, 120, depth=False)
        self.r_cam = self.r_cam[::-1, :, ::-1]

        self.l_render.render(160, 120, camera_id=3)
        self.l_cam = self.l_render.read_pixels(160, 120, depth=False)
        self.l_cam = self.l_cam[::-1, :, ::-1]

    def filtered_camera(self):
        self.render_camera() 

        r_rgb = np.mean(self.r_cam, axis=2)
        l_rgb = np.mean(self.l_cam, axis=2)

        self.r_filter = np.uint8((r_rgb >= 200)*255)
        self.l_filter = np.uint8((l_rgb >= 200)*255)

    def visualize_camera(self, flag):   
        self.filtered_camera()  
        if flag:
            # cv2.imshow('free_cam', free_cam)
            cv2.imshow('r_cam', self.r_cam)
            cv2.imshow('l_cam', self.l_cam)
        else:
            cv2.imshow('r_cam', self.r_filter)
            cv2.imshow('l_cam', self.l_filter)
        cv2.waitKey(1)



    def update_reward_proportionals(self, new_proportionals):
        assert type(self.reward_proportionals) == dict, " Bad Reward Proportionals: ".format(self.reward_proportionals)
        for key in self.reward_proportionals.keys():
            if key in new_proportionals.keys():
                self.reward_proportionals[key] = new_proportionals[key]
        temp = np.array([key in self.reward_proportionals.keys() for key in new_proportionals.keys()])
        # print(" \n\n\n", temp, np.invert(temp), np.array(list(new_proportionals.keys())),"\n\n")
        ign = np.array(list(new_proportionals.keys()))[np.invert(temp)]
        if not temp.all():
            logger.log(f"Update Proportionals: Some of the received keys are not known and will be ignored. (Ignored: {ign})")