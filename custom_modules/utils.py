import numpy as np
from scipy.spatial.transform import Rotation
from garage import EpisodeBatch, StepType

def toEpisodeBatch(env, episode):
    episode = EpisodeBatch(env_spec=env.spec,
                           episode_infos=episode["episode_infos"],
                           observations=np.asarray(episode["observations"]),
                           last_observations=np.asarray([episode["observations"][-1]]),
                           actions=np.asarray(episode["actions"]),
                           rewards=np.asarray(episode["rewards"]),
                           step_types=np.asarray(episode["dones"], dtype=StepType),
                           env_infos=dict(episode["env_infos"]),
                           agent_infos=dict(episode["agent_infos"]),
                           lengths=np.asarray([len(episode["actions"])], dtype='i'))
    return episode

def get_smallest_angle(angle):
    return np.arctan2(np.sin(angle),np.cos(angle))

def get_theta_from_quat(quat):
    if (quat == 0).all():
        return 0.
    [x, _, _] = Rotation.from_quat(quat).as_euler("xyz", degrees=False)
    theta = (np.pi - x) - np.pi/2
    theta = get_smallest_angle(theta)
    return np.round(theta, 4)

#
# Returns initial distance (linear and angular) and
# initial optimal direction to follow (no obstacles considered).
#
def get_init_info(goal_pose, init_pose):
    # init_dist = np.round(np.sqrt((goal_pose[0]-init_pose[0])**2 + (goal_pose[1]-init_pose[1])**2 + (goal_pose[2]-init_pose[2])**2), 4)
    init_dist = np.round(np.sqrt((goal_pose[0]-init_pose[0])**2 + (goal_pose[1]-init_pose[1])**2), 4)
    init_theta_dist = abs(get_smallest_angle(goal_pose[2] - init_pose[2]))
    init_mu = np.arctan2((goal_pose[1] - init_pose[1]), (goal_pose[0] - init_pose[0]))/np.pi
    init_ideal_wheel = get_smallest_angle(init_mu*np.pi - init_pose[2])/np.pi
    return init_dist, init_theta_dist, init_mu, init_ideal_wheel
