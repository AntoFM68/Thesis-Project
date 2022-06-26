#!/usr/bin/env python3
from garage import wrap_experiment, rollout
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.policies import CategoricalCNNPolicy
from garage.trainer import TFTrainer, TrainArgs
import dowel
from dowel import logger, tabular
import numpy as np
import tensorflow as tf
import gc
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from custom_modules.utils import toEpisodeBatch
from custom_envs.mujoco_envs.envs.deepracer_discrete_real import Deepracer_discrete_real # Change file


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

@wrap_experiment
def ppo_deepracer_discrete(ctxt=None, seed=1, resume=True): # Change resume
    
    snapshot_dir = "/root/rl_lab/checkpoints/PPO_Deepracer_discrete5/test" # Change dir
    if not resume:
        # Incremental Directory Names
        i = 0
        while os.path.exists(snapshot_dir+"_%s" %i): i+=1
        snapshot_dir = snapshot_dir+"_"+str(i)
    
    logger.add_output(dowel.TensorBoardOutput(snapshot_dir))
    ctxt.snapshot_dir = snapshot_dir
    with TFTrainer(ctxt) as trainer:
        set_seed(seed)

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
            env = GymEnv('Deepracer_discrete_real-v0')  # Change id

            policy = CategoricalCNNPolicy(env_spec=env.spec,
                filters=((1, (3, 3)), 
                         (8, (1, 1)), 
                         (8, (3, 3)), 
                        (16, (1, 1)),
                        (16, (3, 3)),
                        (32, (1, 1)),
                        (32, (3, 3)),
                         (4, (1, 1))),
                strides=(2, 1, 2, 1, 2, 1, 2, 1),
                padding='SAME',
                hidden_sizes=(320, ))  # yapf: disable

            baseline = GaussianCNNBaseline(env_spec=env.spec,
                filters=((1, (3, 3)), 
                         (8, (1, 1)), 
                         (8, (3, 3)), 
                        (16, (1, 1)),
                        (16, (3, 3)),
                        (32, (1, 1)),
                        (32, (3, 3)),
                         (4, (1, 1))),
                strides=(2, 1, 2, 1, 2, 1, 2, 1),
                padding='SAME',
                hidden_sizes=(320, ),
                use_trust_region=True)  # yapf: disable

            algo = PPO(env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                sampler=None,
                discount=0.97,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=0.05,
                center_adv=False,
                optimizer_args=dict(
                    batch_size=30,
                    max_optimization_epochs=50,
                    learning_rate=0.00001
                ))

            trainer.setup(algo, env)

        for i in range(starting_epoch, 2000):
            episode = rollout(env, trainer._algo.policy, animated=True)
            episode = toEpisodeBatch(env, episode)  # Cast the list to EpisodeBatch
            info = episode.env_infos
            logger.log('Episode: {} | Eval: {} | Reward: {} | Final Distance: {:3.3}'.format(
                i, info["outcome"][-1], np.sum(episode.rewards), info["DIST"][-1]))
            logger.log(f'Training Started')
            # Note: "_train_once" directly stores the episode in the replay buffer
            trainer._algo._train_once(i, episode)
            tabular.record("Env/Reward", np.sum(episode.rewards))
            tabular.record("Env/Dist", info["DIST"][-1])
            logger.log(tabular)
            logger.dump_all()

            if (i+1) % 10 == 0:
                # Make sure that trainer has the most updated versions of algo and env.
                trainer.setup(trainer._algo, env)
                # Save arguments for restore
                trainer._train_args = TrainArgs(n_epochs=i,
                                            batch_size=30,
                                            plot=False,
                                            store_episodes=False,
                                            pause_for_plot=False,
                                            start_epoch=0)
                trainer.save(epoch = i)

            gc.collect()

        logger.log(" Exiting")
        exit()

ppo_deepracer_discrete()