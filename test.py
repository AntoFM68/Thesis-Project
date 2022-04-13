from garage.envs import GymEnv
import custom_envs.mujoco_envs.envs.deepracer


if __name__ == "__main__":

    env_id = "Deepracer-v0"

    env = GymEnv(env_id)
    print("Initialized", type(env).__name__)

    s, i = env.reset()
    # env.visualize()

    ep_step = []
    # ep_step.append(env.step(env.action_space.sample()))
    ep_step.append(env.step([1, 0]))

    print(" EP STEP: ", ep_step[-1].last)
    
    while not ep_step[-1].last:
        env.visualize_camera(True)
        # ep_step.append(env.step(env.action_space.sample()))
        ep_step.append(env.step([1, -0.15]))

        if ep_step[-1].last:
            break

    env.close()
    input("Closing")
