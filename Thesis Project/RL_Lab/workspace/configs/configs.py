import numpy as np
import argparse


SUCCESS = 1
FAIL = 0
SERIOUS_FAIL = -1
EVAL_MSGS = {SUCCESS :      "SUCCESS", 
             FAIL:          "FAIL",
             SERIOUS_FAIL:  "SERIOUS FAIL"}

def get_params():
    parser = argparse.ArgumentParser(description='General_Params')
    parser.add_argument('--CKPT_RATE',            type=int,   default=10,                   help='Checkpoint Rate')
    parser.add_argument('--CKPT_PATH',            type=str,   default="checkpoints",        help='Path for Logs')
    parser.add_argument('--TEST_NAME',            type=str,   default="test",               help='Name of Experiment')
    parser.add_argument('--RESUME',               type=bool,  default=0,                    help='If 1, tries to resume the training from CKPT_PATH/<algo>/TEST_NAME dir.')
    parser.add_argument('--IS_TEST',              type=bool,  default=0,                    help='Whether is a Test (1) or a Training (0, default)')
    parser.add_argument('--MODEL_N',              type=int,   default=0,                    help='# of the model to be tested. If 0, then all saved models will be tested')
    parser.add_argument('--N_EVAL_GOALS',         type=int,   default=100,                  help='Number of Goals to evaluate a good model')

    # parser.add_argument('--VERBOSE',              type=int,   default=1,                    help='Whether to print intermediate and final recap or not.')
    # parser.add_argument('--PRINT_RATE',           type=int,   default=100,                  help='Every PRINT_RATE steps, a short recap of the ongoing episode will be shown.')
    parser.add_argument('--GUI_ENABLED',          type=int,   default=0,                    help='If enabled, Gazebo GUI will be shown.')

    parser.add_argument('--MAX_EPISODES',         type=int,   default=2000,                 help='# of Episodes')
    parser.add_argument('--STEPS_PER_EPOCH',      type=float, default=1,                    help='# of Epochs of Training per Episode. (See Garage Docs)')
    parser.add_argument('--N_TRAIN_STEPS',        type=float, default=1500,                 help='# of Training Steps per Training Epoch. (See Garage Docs)')


#     parser = argparse.ArgumentParser(description='TD3_Params')
    parser.add_argument('--LR_A',                 type=int,   default=0.0001,       help='Learning Rate for Actor')
    parser.add_argument('--LR_C',                 type=int,   default=0.0001,       help='Learning Rate for Critic')
    parser.add_argument('--A_HIDDEN_LAYERS',      nargs='+',  default=[512,256],    help="Actor's hidden layers")
    parser.add_argument('--C_HIDDEN_LAYERS',      nargs='+',  default=[512,256],    help="Critic's hidden layers")
    parser.add_argument('--TAU',                  type=float, default=0.01,         help='Soft Upgrade for Target Nets')
    parser.add_argument('--DISCOUNT',             type=float, default=0.97,         help='Discount Reward')
    parser.add_argument('--ACTOR_UPDATE_PERIOD',  type=float, default=125,          help='Rate at which update Actor Net and apply soft update')


#     parser = argparse.ArgumentParser(description='ReplayBuffer_Params')
    parser.add_argument('--MEMORY_CAPACITY',      type=int,   default=10000,        help='Capacity of Memory')
    parser.add_argument('--MIN_MEM_SIZE',         type=int,   default=300,          help='Minimum # of tuples to access a train step')
    parser.add_argument('--BATCH_SIZE',           type=int,   default=30,           help='Dimension of Buffer Replay')


#     parser = argparse.ArgumentParser(description='Noise_Params')
    parser.add_argument('--EPSILON',              type=float, default=0.3,          help='Exploration Control')
    parser.add_argument('--EPSILON_DECAY',        type=float, default=0.99,         help='Decay of Exploration')


#     parser = argparse.ArgumentParser(description='Reward_Params')
    parser.add_argument('--DONE_REWARD',          type=float, default=1000,         help='Reward given only if EPS_DIST is reached')
    parser.add_argument('--DIST',                 type=float, default=1,            help='Proportional for the distance component of the reward')
    parser.add_argument('--OUT_TRACK',            type=float, default=0.001,        help='Proportional for the distance component on track of the reward')
    # parser.add_argument('--LIMIT',                type=float, default=0.1,          help='Proportional for the distance component on track border of the reward')
    parser.add_argument('--POS_VEL',              type=float, default=1.2,          help='Proportional for the positive velocity of the reward')
    parser.add_argument('--NEG_VEL',              type=float, default=0.1,          help='Proportional for the negative velocity of the reward')
    parser.add_argument('--STEERING',             type=float, default=0.5,          help='Proportional for the over steering of the reward')

    args = parser.parse_args()
    params = vars(args)

    gen_params = {
        "CKPT_RATE": params["CKPT_RATE"],
        "CKPT_PATH": params["CKPT_PATH"],
        "TEST_NAME": params["TEST_NAME"],
        "RESUME": params["RESUME"],
        "IS_TEST": params["IS_TEST"],
        "MODEL_N": params["MODEL_N"],
        "N_EVAL_GOALS": params["N_EVAL_GOALS"],
        # "VERBOSE": params["VERBOSE"],
        # "PRINT_RATE": params["PRINT_RATE"],
        "GUI_ENABLED": params["GUI_ENABLED"],
        "MAX_EPISODES": params["MAX_EPISODES"],
        "STEPS_PER_EPOCH": params["STEPS_PER_EPOCH"],
        "N_TRAIN_STEPS": params["N_TRAIN_STEPS"],
    }

    td3_params = {
        "LR_A": params["LR_A"],
        "LR_C": params["LR_C"],
        "A_HIDDEN_LAYERS": params["A_HIDDEN_LAYERS"],
        "C_HIDDEN_LAYERS": params["C_HIDDEN_LAYERS"],
        "TAU": params["TAU"],
        "DISCOUNT": params["DISCOUNT"],
        "ACTOR_UPDATE_PERIOD": params["ACTOR_UPDATE_PERIOD"],
    }
    
    buffer_params = {
        "MEMORY_CAPACITY": params["MEMORY_CAPACITY"],
        "MIN_MEM_SIZE": params["MIN_MEM_SIZE"],
        "BATCH_SIZE": params["BATCH_SIZE"],
    }
    
    epnoise_params = {
        "EPSILON": params["EPSILON"],
        "EPSILON_DECAY": params["EPSILON_DECAY"],
    }
    
    reward_params = {
        "DONE_REWARD": params["DONE_REWARD"],
        "DIST": params["DIST"],
        "OUT_TRACK": params["OUT_TRACK"],
        # "LIMIT": params["LIMIT"],
        "POS_VEL": params["POS_VEL"],
        "NEG_VEL": params["NEG_VEL"],
        "STEERING": params["STEERING"],
    }

    return gen_params, td3_params, buffer_params, epnoise_params, reward_params
