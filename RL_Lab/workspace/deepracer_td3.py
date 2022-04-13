import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from garage import wrap_experiment, rollout
from garage.envs import GymEnv
from garage.experiment import deterministic
from garage.tf.algos import TD3
from garage.trainer import TFTrainer, TrainArgs
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.np.exploration_policies import AddGaussianNoise
from garage.replay_buffer import PathBuffer

import numpy as np
import tensorflow as tf
import time
import dowel
from dowel import logger, tabular

from custom_envs.mujoco_envs.envs.deepracer import Deepracer
from custom_models.continuous_cnnmlp_policy import ContinuousCnnMlpPolicy

from custom_modules.utils import toEpisodeBatch
from configs.configs import get_params, \
                            EVAL_MSGS, SUCCESS, FAIL, SERIOUS_FAIL
                            
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


@wrap_experiment
def td3_deepracer(ctxt, env_id, seed=0, kwargs=None):

    general_params = kwargs["general_params"]
    buffer_params = kwargs["buffer_params"]
    reward_params = kwargs["reward_params"]

    snapshot_dir = "/root/rl_lab/"+general_params["CKPT_PATH"]+"/TD3_Deepracer/"+general_params["TEST_NAME"]
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
            env = GymEnv(env_id)
            env.update_reward_proportionals(reward_params)

            policy = ContinuousCnnMlpPolicy(
                env_spec=env.spec,
                input_dim_cnn=(120, 320, 3),
                filters_cnn=((3, (3, 5)), (32, (3, 3)), (5, (1, 1))),
                strides_cnn=(1, 2, 1),
            )

            qf = ContinuousMLPQFunction(
                name='ContinuousMLPQFunction',
                env_spec=env.spec,
                hidden_sizes=[256,256],
                action_merge_layer=0,
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=None,
            )

            qf2 = ContinuousMLPQFunction(
                name='ContinuousMLPQFunction2',
                env_spec=env.spec,
                hidden_sizes=[256,256],
                action_merge_layer=0,
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=None,
            )

            exploration_policy = AddGaussianNoise(
                env.spec,
                policy,
                total_timesteps=2000,
                max_sigma=0.1,
                min_sigma=1e-16,
                decay_ratio=50,
            )

            replay_buffer = PathBuffer(
                capacity_in_transitions=5000)


            algo = TD3(env.spec,
                policy=policy,
                qf=qf,
                qf2=qf2,
                replay_buffer=replay_buffer,
                sampler=None,
                steps_per_epoch=1,
                policy_lr=0.0001,
                qf_lr=0.0001,
                target_update_tau=0.005,
                n_train_steps=500,
                discount=0.97,
                min_buffer_size=300,
                buffer_batch_size=30,
                exploration_policy=exploration_policy,
                policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                qf_optimizer=tf.compat.v1.train.AdamOptimizer,
            )

            trainer.setup(algo, env)

        for i in range(starting_epoch, general_params['MAX_EPISODES']):
            episode = rollout(env,
                            trainer._algo.exploration_policy,
                            animated=False#(((i+1) % 25)==0),
                            )
            episode = toEpisodeBatch(env, episode)  # Cast the episode to EpisodeBatch
            trainer._algo.exploration_policy.update(episode)
            info = episode.env_infos
            logger.log('Episode: {} | Eval: {} | Final Distance: {:3.3} | Reward: {} '.format(
                i, info["outcome"][-1], info["DIST"][-1], np.sum(episode.rewards)))
            logger.log(f'Training Started')
            # Note: "_train_once" directly stores the episode in the replay buffer
            trainer._algo._train_once(i, episode)
            tabular.record("Env/Reward", np.sum(episode.rewards))
            tabular.record("Env/Dist", info["DIST"][-1])
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

        logger.log(" Exiting")
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

def test_environment():
    env = GymEnv("Deepracer-v0")
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

def main():
    kwargs = dict()
    kwargs["general_params"], \
        kwargs["td3_params"], \
        kwargs["buffer_params"], \
        kwargs["noise_params"], \
        kwargs["reward_params"] = get_params()
    td3_deepracer(seed=0, env_id="Deepracer-v0", kwargs=kwargs)

if __name__ == "__main__":
    main()
    # test_environment()
