from garage.envs import GymEnv
from garage.experiment import Snapshotter
import tensorflow as tf
from custom_envs.mujoco_envs.envs.deepracer_discrete_real import Deepracer_discrete_real
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


env_id = "Deepracer_discrete_real-v0"
env = GymEnv(env_id)
print("Initialized", type(env).__name__)

s, i = env.reset()
env.visualize()
ep_step = []

with tf.compat.v1.Session() as sess:
    data = Snapshotter().load('/root/rl_lab/results')
    policy = data['algo'].policy
    
    ep_step.append(env.step(env.action_space.sample()))

    while not ep_step[-1].last:
        ep_step.append(env.step(policy.get_action(env._env.env.obs_cam)[0]))

        if ep_step[-1].last:
            break

    env.close()
    input("Closing")
