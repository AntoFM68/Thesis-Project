from garage.envs import GymEnv
from custom_envs.mujoco_envs.envs.deepracer_discrete_real import Deepracer_discrete_real

if __name__ == "__main__":

    env_id = "Deepracer_discrete_real-v0"
    env = GymEnv(env_id)
    print("Initialized", type(env).__name__)

    s, i = env.reset()
    env.visualize()

    ep_step = []
    ep_step.append(env.step(env.action_space.sample()))

    print("EP STEP:", ep_step[-1].last)
    
    while not ep_step[-1].last:
        # ep_step.append(env.step(env.action_space.sample()))
        ep_step.append(env.step(12))

        if ep_step[-1].last:
            break

    env.close()
    input("Closing")
