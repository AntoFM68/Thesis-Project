# from . import envs
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Deepracer_discrete-v0',
    entry_point='custom_envs.mujoco_envs.envs:Deepracer_discrete',
    max_episode_steps=2500,
    reward_threshold=100000.0,
    nondeterministic = True,
)

register(
    id='Deepracer_discrete_real-v0',
    entry_point='custom_envs.mujoco_envs.envs:Deepracer_discrete_real',
    max_episode_steps=2500,
    reward_threshold=100000.0,
    nondeterministic = True,
)