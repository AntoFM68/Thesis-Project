# from . import envs
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Turtlebot-v0',
    entry_point='custom_envs.mujoco_envs.envs:Turtlebot',
    # timestep_limit=1500,
    max_episode_steps=1000,
    reward_threshold=1000.0,     # Unreachable (in our case should be)
    nondeterministic = True,
)

register(
    id='Deepracer-v0',
    entry_point='custom_envs.mujoco_envs.envs:Deepracer',
    # timestep_limit=1500,
    max_episode_steps=1000,
    reward_threshold=1000.0,     # Unreachable (in our case should be)
    nondeterministic = True,
)
