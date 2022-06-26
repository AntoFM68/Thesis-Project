import os
import re
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from custom_modules.utils import get_theta_from_quat
from mujoco_py import cymj, MjRenderContext
from dowel import logger
import cv2
from random import randrange
import copy
import imageio


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

    V = np.sqrt(vels[0]**2 + vels[1]**2)*np.sign(velocimeter[0])
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
standard_flag = False
camera_flag = True
visual_flag = True

class Deepracer_discrete_real(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.main_body = "aws_deepracer"
        self.image_w = 160
        self.image_h = 120
        self.state_shape = (self.image_h, self.image_w, 1)   # Should be specified for every new environment.
        self.action_shape = 2  # Should be specified for every new environment.
        self.action_sim = [0., 0.]
        self.default_pose = [0., 0., 0.]

        self.reset_pose = {0: [10.,   -11.47, 1.,  0., 0., 0. ],  # Medium   BOWTIE
                           1: [ 7.,   -11.47, 0.,  0., 0., 1. ],  # M
                           2: [ 7.,    -7.65, 1.,  0., 0., 0. ],  # Easy
                           3: [10.,    -7.65, 0.,  0., 0., 1. ],  # E
                           4: [15.98, -10.3,  0.7, 0., 0., 0.7],  # E
                           5: [ 0.83,  -9.,   0.7, 0., 0., 0.7],  # M
                           6: [ 5.18,  10.,   0.7, 0., 0., 0.7],  # M        OVAL
                           7: [ 5.18,   5.2,  0.7, 0., 0., 0.7],  # E
                           8: [ 5.18,  10.,  -0.7, 0., 0., 0.7],  # E
                           9: [ 5.18,   5.2, -0.7, 0., 0., 0.7],  # M
                          10: [-5.,     3.23, 0.,  0., 0., 1. ],  # E        MONZA
                          11: [-5.,     3.23, 1.,  0., 0., 0. ],  # Hard
                          12: [-7.5,    6.08, 1.,  0., 0., 0. ],  # H
                          13: [-7.5,    6.08, 0.,  0., 0., 1. ],  # M
                          14: [10.,    -4.23, 0.,  0., 0., 1. ],  # E        SHORT
                          15: [11.4,    4.26, 1.,  0., 0., 0. ],  # H
                          16: [11.4,    4.26, 0.,  0., 0., 1. ],  # H
                          17: [11.78,  -1.9,  0.7, 0., 0., 0.7],  # H
                          18: [-4.,     0.48, 0.,  0., 0., 1. ],  # E        LONG
                          19: [-4.,     0.48, 1.,  0., 0., 0. ],  # M
                          20: [-8.3,   -1.39, 1.,  0., 0., 0. ],  # H
                          21: [-8.3,   -1.39, 0.,  0., 0., 1  ]}  # H
        self.dist = 0
        self.step_size = 0.01

        self.r_cam = np.zeros((self.image_h, self.image_w, 3), dtype=np.float32)
        self.l_cam = np.zeros((self.image_h, self.image_w, 3), dtype=np.float32)
        self.obs_cam = np.zeros((self.image_h, self.image_w, 1), dtype=np.float32)
        self.gif = []
        self.init_flag = False

        # self.configs = get_general_params()
        self.nstep = 0
        # self.max_steps = kwargs["N_TRAIN_STEPS"]
        self.reward_proportionals = {
            "OUT_TRACK": 0.001,
            "LIMIT": 0.8,
            "POS_VEL": 1.4,
            "STEERING": 0.5,
            "ZIGZAG": 1.2
        }
        assert type(self.reward_proportionals) == dict, " Bad Reward Proportionals: {}".format(type(self.reward_proportionals))

        self.info = None

        self.exec_mode = _TRAIN_MODE

        self.init_pose = self.default_pose
        self.previous_pose = self.init_pose
        self.pose = self.init_pose

        # Note: if model_path starts with "/", it will be considered as the full path.
        #       otherwise, it will be considered relative to "assets" folder (in the gym path installation).
        #       If not found, will raise an error.
        assests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        mujoco_env.MujocoEnv.__init__(self, model_path=os.path.join(assests_dir,"sharp_deepracer_real.xml"), frame_skip=33)
        # cymj.set_pid_control(self.model, self.data) # FROM: https://github.com/openai/mujoco-py/blob/master/mujoco_py/tests/test_pid.py
        utils.EzPickle.__init__(self)
        
        self.observation_space = self._set_observation_space()
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Flag to understand when an episode has reached an end. (Used to call the evaluation of the episode)
        self._ep_fin = False
        
        if visual_flag:
            self.visual_window = MjRenderContext(sim=self.sim, device_id=1, offscreen=True, opengl_backend='glfw')
        self.r_render = MjRenderContext(sim=self.sim, device_id=2, offscreen=True, opengl_backend='glfw')
        self.l_render = MjRenderContext(sim=self.sim, device_id=3, offscreen=True, opengl_backend='glfw')
        self.sim.add_render_context(self.r_render)
        self.sim.add_render_context(self.l_render)
        self.init_flag = True

    def _set_observation_space(self, ob=None):
        observation_space = spaces.Box(
            low=0, high=255, shape=(self.image_h, self.image_w, 1), dtype=np.float32
        )
        return observation_space

    def _set_action_space(self):
        self.discrete_throttle = [0.4, 0.6, 0.8, 1.,
                                  0.4, 0.6, 0.8, 1.,
                                  0.4, 0.6, 0.8, 1.,
                                  0.4, 0.6, 0.8, 1.,
                                  0.4, 0.6, 0.8, 1.,
                                  0.4, 0.6, 0.8, 1.,
                                  0.4, 0.6, 0.8, 1.,]
        self.discrete_steering = [-0.523599, -0.523599, -0.523599, -0.523599,  # -30°
                                  -0.349066, -0.349066, -0.349066, -0.349066,  # -20°
                                  -0.174533, -0.174533, -0.174533, -0.174533,  # -10°
                                   0.,        0.,        0.,        0.,        #   0°
                                   0.174533,  0.174533,  0.174533,  0.174533,  #  10°
                                   0.349066,  0.349066,  0.349066,  0.349066,  #  20°
                                   0.523599,  0.523599,  0.523599,  0.523599,] #  30°
        self.action_space = spaces.Discrete(28)
        return self.action_space

    def reset_model(self):
        if hasattr(self, "sim"):
            self.r_render = MjRenderContext(sim=self.sim, device_id=2, offscreen=True, opengl_backend='glfw')
            self.l_render = MjRenderContext(sim=self.sim, device_id=3, offscreen=True, opengl_backend='glfw')

        self._n_step = 0
        self.nstep = 0
        self.dist = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        random_p = randrange(22)
        qpos[0:2] = self.reset_pose[random_p][0:2]
        qpos[3:7] = self.reset_pose[random_p][2:6]

        self.set_state(qpos, qvel)  # MujocoEnv default func
        self.init_pose = copy.deepcopy(self.sim.data.get_body_xpos(self.main_body))
        self.init_pose[2] = copy.deepcopy(get_theta_from_quat(self.sim.data.get_body_xquat(self.main_body)))
        self.pose = self.init_pose
        self.previous_pose = self.init_pose
        if random_p >= 0 and random_p < 6:
            print('     ***     Reset pose on BOWTIE circuit in point {} with pose: {}'.format(random_p+1, self.init_pose))
        if random_p >= 6 and random_p < 10:
            print('     ***     Reset pose on OVAL circuit in point {} with pose: {}'.format(random_p+1, self.init_pose))
        if random_p >= 10 and random_p < 14:
            print('     ***     Reset pose on MONZA circuit in point {} with pose: {}'.format(random_p+1, self.init_pose))
        if random_p >= 14 and random_p < 18:
            print('     ***     Reset pose on SHORT circuit in point {} with pose: {}'.format(random_p+1, self.init_pose))
        if random_p >= 18 and random_p < 22:
            print('     ***     Reset pose on LONG circuit in point {} with pose: {}'.format(random_p+1, self.init_pose))

        ob = self._get_obs()
        return ob

    def step(self, action):
        self.previous_steering = copy.deepcopy(self.action_sim[1])
        self.action_sim[0] = self.discrete_throttle[action]
        self.action_sim[1] = self.discrete_steering[action]
        
        # Get Current State | Apply Action | # Get New State

        self.do_simulation(self.action_sim, self.frame_skip)
        self.nstep += 1

        obs = self._get_obs()
        reward, track, more_info = self._reward()

        done = self._done_cond(track)
        win = self.win()

        # PRINT Section
        # print('  *** ACTION:', self.action)
        # print('  *** REWARD:', reward)
        # print('  *** POSE:', self.pose)
        # print('  *** DIST:', self.dist)
        # print('  *** VELS:', self.info["AVG_REAL_VEL_INFO"])
        # print('  *** SENS VELS:', self.info["SENS_VEL"])

        # Make gif
        # if win or done or self.nstep > 1000:
        #     self.make_gif()

        # Add some more info here
        more_info["outcome"] = "Success" if win else "Failure"
        more_info["DIST"] = self.dist
                
        return (obs, reward, done, more_info)
    
    def _get_obs(self):
        robot = self.sim.data
        self.info = gather_info_mujoco(robot, self.previous_pose, main_body=self.main_body)
        self.dist += self.info["DIST"]
        self.pose = self.info["POSE"]
        self.previous_pose = copy.deepcopy(self.pose)

        if self.init_flag:
            self.render_camera()
        ob = self.obs_cam

        assert ob.shape == self.state_shape, "Observation is {} but should be ({},) instead.".format(ob.shape, self.state_shape)
        return ob

    def _reward(self):
        obs_mean = np.mean(self.obs_cam[100:120], axis=0)
        r_track = np.mean(obs_mean[100:145])
        l_track = np.mean(obs_mean[15:60])
        track = r_track + l_track
        r_limit = np.mean(obs_mean[80:100])
        l_limit = np.mean(obs_mean[60:80])
        limit = r_limit + l_limit

        reward = 1.
        if self.info["AVG_REAL_VEL_INFO"][0] > 0.6:
            reward *= self.reward_proportionals["POS_VEL"]
        if abs(self.action_sim[1]) > 0.349066: # 20°
            reward *= self.reward_proportionals["STEERING"]
        if abs(self.action_sim[1] - self.previous_steering) <= 0.174533:
            reward *= self.reward_proportionals["ZIGZAG"]
        if track > 0:
            reward *= self.reward_proportionals["LIMIT"]
        if limit > 0:
            reward *= self.reward_proportionals["OUT_TRACK"]
        
        more_info = {}

        return reward, limit, more_info

    def _done_cond(self, track):
        if track > 160:
            print('     ***     Failure! Out of the track.')
            print('     ***     Final distance:', self.dist, ' | Final pose:', self.pose)
            return True
        else:
            return False
    
    def win(self):
        if self.nstep > 2499:
            print('     ***     Finish! No track limits exceeded.')
            print('     ***     Final distance:', self.dist, ' | Final pose:', self.pose)
            return True

    def set_test_mode(self):
        self.exec_mode = _TEST_MODE        # TEST = False | TRAIN = True

    def set_train_mode(self):    # Redundant: Training mode is default
        self.exec_mode = _TRAIN_MODE


    def render(self, mode):
        if standard_flag:
            super().render(mode)
        if camera_flag:
            self.visualize_camera()
        if visual_flag:
            self.visual_window.render(400, 300, camera_id=1) # -1, 1
            visual = self.visual_window.read_pixels(400, 300, depth=False)
            visual = visual[::-1, :, ::-1]
            # Make gif
            # self.gif.append(visual[:, :, ::-1])
            cv2.imshow('visual', visual)
            cv2.waitKey(1)

    def render_camera(self):
        self.r_render.render(160, 120, camera_id=2)
        self.r_cam = self.r_render.read_pixels(160, 120, depth=False)
        self.r_cam = self.r_cam[::-1, :, ::-1]

        self.l_render.render(160, 120, camera_id=3)
        self.l_cam = self.l_render.read_pixels(160, 120, depth=False)
        self.l_cam = self.l_cam[::-1, :, ::-1]

        obs_rgb = np.hstack((self.l_cam[:, 0:80, :], self.r_cam[:, 80:160, :]))
        # Make gif
        # self.gif.append(self.obs_cam[:, :, ::-1])

        self.obs_cam = cv2.cvtColor(obs_rgb, cv2.COLOR_BGR2GRAY)
        self.obs_cam[:40] = 150
        self.obs_cam[40:] = np.float32((self.obs_cam[40:] >= 200)*255)
        self.obs_cam = np.reshape(self.obs_cam, [self.image_h, self.image_w, 1])

    def visualize_camera(self):
        cv2.imshow('R+L Cam', self.obs_cam)
        cv2.waitKey(1)

    def make_gif(self):
        print('  Saving gif...')
        imageio.mimsave('../results/Long.gif', self.gif, duration=1/30)
        print('  Gif saved!')
        input()
