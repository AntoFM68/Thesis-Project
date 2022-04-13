from custom_envs.mujoco_envs.envs.turtlebot import Turtlebot
from custom_modules.utils import toEpisodeBatch
from configs.configs import get_params, \
                            EVAL_MSGS, SUCCESS, FAIL, SERIOUS_FAIL

import numpy as np
import tensorflow as tf
import gym
# from gym import logger as log
from garage.envs import GymEnv, normalize
from garage import wrap_experiment, rollout
from garage.experiment import deterministic
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker, VecWorker, LocalSampler, WorkerFactory, MultiprocessingSampler, RaySampler
from garage.tf.algos import PPO
from garage.trainer import TFTrainer, TrainArgs
import dowel
import os
from dowel import logger, tabular
import time
from threading import Thread

from garage.tf.policies import GaussianMLPPolicy
from garage.tf.baselines import GaussianMLPBaseline

# log.set_level(log.INFO)

#####################################################################################
#
# Notes:
# - The evaluation is implemented in garage/_functions.py/obtain_evaluation_episodes()
#   This function cannot be controlled directly, however, it is called in the "train"
#   function of each algorithm. Either change the default value of the parameter of the 
#   obtain_evaluation_episodes() function or add in the algorithm the desired values. 
#####################################################################################


@wrap_experiment
def custom_train(ctxt, seed=0, kwargs=None):

    general_params = kwargs["general_params"]
    buffer_params = kwargs["buffer_params"]
    reward_params = kwargs["reward_params"]
    noise_params = kwargs["noise_params"]

    snapshot_dir = "/root/rl_lab/"+general_params["CKPT_PATH"]+"/PPO/"+general_params["TEST_NAME"]
    if not general_params["RESUME"]:
        # Incremental Directory Names
        i = 0
        while os.path.exists(snapshot_dir+"_%s" %i): i+=1
        snapshot_dir = snapshot_dir+"_"+str(i)
    
    logger.add_output(dowel.TensorBoardOutput(snapshot_dir))
    ctxt.snapshot_dir = snapshot_dir
    with TFTrainer(ctxt) as trainer:
        # print("SNAPSHOT DIR: ", snapshot_dir)
        # input()
        deterministic.set_seed(seed)

        restored = False
        try:
            trainer.restore(snapshot_dir)
            algo = trainer._algo
            env = trainer._env
            restored = True
            starting_epoch = trainer._train_args.n_epochs+1
            logger.log("Environment Restored Successfully.")
        except Exception as e:
            restored = False
            print(" \n\n Exception: ", e)
            logger.log("Warning: Could not restore the environment. A new training will be started.")

        if not restored:
            starting_epoch = 0
            env = GymEnv("Turtlebot-v0")
            env.update_reward_proportionals(reward_params)

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(512,256),
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.tanh,
            )

            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                hidden_sizes=(512,256),
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=None,
                use_trust_region=True,
            )

            exploration_policy = AddGaussianNoise(
                env.spec,
                policy,
                total_timesteps=general_params['MAX_EPISODES'] * general_params['STEPS_PER_EPOCH'],    # If < max_ep_steps*n_episodes then reaches 0 before end of training
                max_sigma=noise_params['EPSILON'],
                min_sigma=-noise_params['EPSILON'],
                decay_ratio=noise_params['EPSILON_DECAY'])

            # replay_buffer = PathBuffer(capacity_in_transitions=buffer_params['MEMORY_CAPACITY'])
            # replay_buffer = PER(capacity_in_transitions=buffer_params['MEMORY_CAPACITY'], env_spec=env.spec)

            # Used to implement Evaluation
            worker = VecWorker(
                seed=0,
                max_episode_length=env.spec.max_episode_length,
                worker_number=1,
                n_envs=100,
                )

            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                sampler=None,
                discount=0.97,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                optimizer_args=dict(
                    learning_rate=0.0001,
                    batch_size=2024,
                    max_optimization_epochs=50,
                ),
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=0.05,
                center_adv=False,
            )

            trainer.setup(algo, env)

        for i in range(starting_epoch, general_params['MAX_EPISODES']):
            # exploration_policy.policy = trainer._algo.policy
            episode = rollout(env,
                            trainer._algo.policy,
                            animated=False#(((i+1) % 25)==0),
                            )
            episode = toEpisodeBatch(env, episode)  # Cast the list to EpisodeBatch
            # exploration_policy.update(episode)
            info = episode.env_infos
            logger.log('Episode: {} - Eval: {} - Final Distance: {:3.3} - Reward: {} '.format(
                i, info["outcome"][-1], info["lin_dist"][-1], np.sum(episode.rewards)))
            # logger.log(f'Training Started')
            # Note: "_train_once" directly stores the episode in the replay buffer
            trainer._algo._train_once(i, episode)
            tabular.record("Env/Reward", np.sum(episode.rewards))
            tabular.record("Env/LinearDist", info["lin_dist"][-1])
            logger.log(tabular)
            logger.dump_all()

            if ((i+1) % general_params['CKPT_RATE']) == 0:
                # Make sure that trainer has the most updated versions of algo and env.
                trainer.setup(trainer._algo, env)
                # Save arguments for restore
                trainer._train_args = TrainArgs(n_epochs=i,
                                            batch_size=buffer_params['BATCH_SIZE'],
                                            plot=False,
                                            store_episodes=False,
                                            pause_for_plot=False,
                                            start_epoch=0)
                trainer.save(epoch = i)

        print(" Exiting")
        exit()

def evaluate(worker, env, policy):
    worker.update_env(env)
    worker.update_agent(policy)
    t = time.time()
    episodes = worker.rollout()
    results = np.array(episodes.env_infos["outcome"][np.cumsum(episodes.lengths)-1])
    elapsed = time.time()-t
    successful = np.sum(results == "Success")
    logger.log(f"Evaluation Results: {successful}/100 - Elapsed: {elapsed}")

def main():
    kwargs = dict()
    kwargs["general_params"], \
        kwargs["td3_params"], \
        kwargs["buffer_params"], \
        kwargs["noise_params"], \
        kwargs["reward_params"] = get_params()
    # td3_garage_tf(seed=0, kwargs=kwargs)
    custom_train(seed=0, kwargs=kwargs)


def test_environment():
    env = GymEnv("Turtlebot-v0")
    s, i = env.reset()
    env.visualize()
    act = env.action_space.sample()
    env_step = env.step(act)

    while not env_step.last:
        act = env.action_space.sample()
        env_step = env.step(act)
        if env_step.last: break

    env.close()
    print(" OUTCOME: ", env_step.env_info["outcome"])
    print("Closing ")



if __name__ == "__main__":
    main()
    # test_environment()
