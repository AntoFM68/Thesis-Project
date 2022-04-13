import os
import numpy as np
from numpy.core.records import array
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from custom_modules.utils import get_theta_from_quat, get_smallest_angle, get_init_info

from mujoco_py import cymj

from dowel import logger



def random_sample_goal(init_pose, min_dist=0.25, max_x=3.5, max_y=3.5, round=4):
    goal_pose = [0,0,0]
    while np.linalg.norm(goal_pose[:2]-init_pose[:2]) < min_dist:
        goal_pose = np.round([
            np.random.uniform(-max_x, max_y),
            np.random.uniform(-max_x, max_y),
            np.random.uniform(-np.pi, np.pi)
        ], round)
    return goal_pose


# Gather Info required for the State
def gather_info_mujoco(robot, goal, main_body):

    # Compute General Info
    pose = robot.get_body_xpos(main_body)
    pose[2] = get_theta_from_quat(robot.get_body_xquat(main_body))
    vels = [*robot.get_body_xvelp(main_body)[:2],    # Xdot, Ydot,
                    robot.get_body_xvelr(main_body)[2]]  # Thetadot
    
    #
    # The indeces of the sensors strictly depends on the order of definition in the .xml file.
    # In this case: accelerometer, velocimiter... 
    #  
    accelerometer = robot.sensordata[0:3]
    velocimeter = robot.sensordata[3:6]

    V = np.sqrt(vels[0]**2 + vels[1]**2)
    Vangle = np.arctan2(vels[1], vels[0])
    
    # Compute Goal-Related info/errors
    dist = np.round(np.sqrt((goal[0] - pose[0])**2 + (goal[1] - pose[1])**2), 4)
    dist_ang = np.arctan2((goal[0] - pose[0]), (goal[1] - pose[1]))
    theta_dist = get_smallest_angle(goal[2] - pose[2])

    info = {
        "POSE": pose,
        "VELS": vels,
        "AVG_REAL_VEL_INFO": [V, Vangle],
        "SENS_ACC": accelerometer,
        "SENS_VEL": velocimeter,
        "DIST": dist,
        "DIST_ANG": dist_ang,
        "THETA_DIST": theta_dist,
    }
    return info

# Compute Reward
def reward_func_mujoco(info, proportionals):

    dist = info["DIST"]

    dist                    = - dist                    *   proportionals["DIST"]
    # err_theta               = - abs(err_theta)          *   proportionals["THETA"]

    comps = np.array([dist])
    cost = np.sum(comps)

    more_info = {
        "rew_dist": comps[0],
        # "rew_err_theta": comps[1],
    }
    return cost, more_info


_TRAIN_MODE = True
_TEST_MODE = False


class Turtlebot(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.main_body = "turtlebot"
        self.state_shape = 6   # Should be specified for every new environment.
        self.action_shape = 2  # Should be specified for every new environment.
        self.default_pose = [0., 0., 0.]
        self.step_size = 0.01

        # self.gather_info = gather_info_mujoco
        # self.reward_func = reward_func_mujoco

        # Select a goal sampling strategy
        self.sample_goal = random_sample_goal

        # Input Parameters
        self.dist_thresh = 0.05
        self.theta_thresh = 0.15
        # self.configs = get_general_params()
        self.nstep = 0
        # self.max_steps = kwargs["N_TRAIN_STEPS"]
        self.reward_proportionals = {
            "DONE_REWARD": 30,
            "DIST": 1,
        }
        assert type(self.reward_proportionals) == dict, " Bad Reward Proportionals: {}".format(type(self.reward_proportionals))

        # Distances Normalized?
        self._NORM_DIST = True
        self._NORM_ANG = True

        self.info = None

        # In Train/Test mode affects the generation of the goals.
        # In Train mode, a goal is sampled using the current goal generation policy (default is random) 
        # In Test mode, you must set a goal manually, using "set_new_goal" method, which requires either a list or numpy array in input.
        # Use "set_test_mode"/"set_train_mode" functions to switch between modalities.
        self.exec_mode = _TRAIN_MODE


        self.init_pose = self.default_pose
        self.pose = self.init_pose
        self.goal_pose = [10,10,np.pi]      # Necessario: in fase di inizalizzazione viene chiamata la funzione "step"
        self.init_dist = 14.142             # in cui si accede a queste variabili.
        self.init_theta_dist = np.pi        #
        self.dist = 14.142                  # Polar coordinate of the goal in the reference frame - Distance
        self.dist_ang = np.pi/4             # Polar coordinate of the goal in the reference frame - Angle
        self.theta_dist = np.pi             #
        # max_point_speed = 0.6 # m/s (should be, documentation lacks)
        # self.mt_per_step = self.step_size * max_point_speed # mt_step = step_s * m/s (0.006)

        # Note: if model_path starts with "/", it will be considered as the full path.
        #       otherwise, it will be considered relative to "assets" folder (in the gym path installation).
        #       If not found, will raise an error.
        assests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        mujoco_env.MujocoEnv.__init__(self, model_path=os.path.join(assests_dir,"turtlebot.xml"), frame_skip=1)
        cymj.set_pid_control(self.model, self.data) # FROM: https://github.com/openai/mujoco-py/blob/master/mujoco_py/tests/test_pid.py
        utils.EzPickle.__init__(self)

        self.observation_space = self._set_observation_space()
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Flag to understand when an episode has reached an end. (Used to call the evaluation of the episode)
        self._ep_fin = False


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

    def _set_observation_space(self, ob=None):
        # For simplicity, bounds are set to be -/+ inf.
        low = np.full(self.state_shape, -float("inf"), dtype=np.float64)
        high = np.full(self.state_shape, float("inf"), dtype=np.float64)
        observation_space = spaces.Box(low, high, dtype=np.float64)
        return observation_space

    def _set_action_space(self):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        high = np.ones(self.action_shape, dtype=np.float64)
        self.action_space = spaces.Box(low=-high, high=high, dtype=np.float64)
        return self.action_space

    def reset_model(self):
        self._n_step = 0
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)  # MujocoEnv default func

        self.init_pose = self.sim.data.get_body_xpos(self.main_body)
        self.init_pose[2] = get_theta_from_quat(self.sim.data.get_body_xquat(self.main_body))
        self.pose = self.init_pose

        if self.exec_mode == _TRAIN_MODE:
            self.goal_pose = self.sample_goal(self.init_pose)
        elif self.exec_mode == _TEST_MODE:
            assert self.goal_pose is not None, \
                " [Reset Task Error]: When in test mode, it is necessary to set manually the new goal."

        self.init_theta_dist = get_smallest_angle(self.goal_pose[2] - self.pose[2])
        self.init_dist = np.round(np.sqrt((self.goal_pose[0] - self.pose[0])**2 + (self.goal_pose[1] - self.pose[1])**2), 4)

        ob = self._get_obs()

        self.init_dist = self.info["DIST"]
        # self.init_mu = self.info["MU"]

        return ob

    def step(self, action):
        # Get Current State | Apply Action | # Get New State
        # Note: the very first step `self.sim.data.get_body_xquat("swervebot")` returns
        # a [0 0 0 0] array. This might be problematic, however, it is managed in the
        # `get_theta_from_quat` function. (It returns a 0 directly)
        # old_pose = [*self.sim.data.get_body_xpos("swervebot")[:2], get_theta_from_quat(self.sim.data.get_body_xquat("swervebot"))]
        # old_vels = [*self.sim.data.get_body_xvelp("swervebot")[:2], self.sim.data.get_body_xvelr("swervebot")[2]]
        # old_pose = self.get_body_com("swervebot")
        self.do_simulation(action, self.frame_skip)
        self.nstep += 1
        # new_pose = [*self.sim.data.get_body_xpos("swervebot")[:2], get_theta_from_quat(self.sim.data.get_body_xquat("swervebot"))]
        # new_vels = [*self.sim.data.get_body_xvelp("swervebot")[:2], self.sim.data.get_body_xvelr("swervebot")[2]]
        # Chech this infos v
        # sim_info = [ self.sim.data.qpos.flat,
        #                 self.sim.data.qvel.flat,
        #                 self.sim.data.cfrc_ext]

        obs = self._get_obs()
        reward, more_info = self._get_reward()
        done = self._done_cond()
        win = self.win()
        if win:
            reward += self.reward_proportionals["DONE_REWARD"]

        # Add some more info here
        more_info["outcome"] = "Success" if win else "Failure"
        more_info["lin_dist"] = self.dist
        # more_info["ang_dist"] = self.err_theta


        return (
            obs,
            reward,
            done,
            more_info
        )

    def _get_reward(self):
        temp_info = self.info
        # Need to set values to current implementation.
        # F.i.: values are modified (normalization).
        temp_info["DIST"] = self.dist
        # temp_info["THETA_DIST"] = self.theta_dist
        return reward_func_mujoco(temp_info, self.reward_proportionals)
    
    def _done_cond(self):
        return (self.dist < self.dist_thresh)# and (abs(self.theta_dist) < self.theta_thresh)
    
    def win(self):
        return (self.dist < self.dist_thresh)# and (abs(self.theta_dist) < self.theta_thresh)

    def _get_obs(self):
        robot = self.sim.data
        self.info = gather_info_mujoco(robot, self.goal_pose, main_body=self.main_body)

        self.pose = self.info["POSE"]
        if self._NORM_DIST: self.dist = self.info["DIST"]/self.init_dist
        else: self.dist = self.info["DIST"]
        self.dist_ang = self.info["DIST_ANG"]

        if self._NORM_ANG: self.err_theta = self.info["THETA_DIST"]/self.init_theta_dist
        else: self.theta_dist = self.info["THETA_DIST"]

        ob = np.hstack((
            self.dist,
            self.dist_ang,
            self.pose[2],
            *self.info["VELS"],   # (X Y THETA)dot
            ))
        assert ob.shape == (self.state_shape,), "Observation is {} but should be ({},) instead.".format(ob.shape, self.state_shape)
        
        return ob


    def set_test_mode(self):
        self.exec_mode = _TEST_MODE        # TEST = False | TRAIN = True

    def set_train_mode(self):    # Redundant: Trainining mode is default
        self.exec_mode = _TRAIN_MODE

    def set_new_goal(self, goal_pose):
        if type(goal_pose) in ["list", "numpy.ndarray"]:
            self.goal_pose = goal_pose
        else:
            raise Exception(f"You must provide either a list or a numpy array. You provided {type(goal_pose)} instead. ")
