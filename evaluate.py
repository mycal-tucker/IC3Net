import sys
import time
import signal
import argparse
import time, os
import numpy as np
import torch
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from evaluator import Evaluator
from args import get_args


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


def load(path):
    # d = torch.load(path)
    # policy_net.load_state_dict(d['policy_net'])

    load_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "models")
    print(f"load directory is {load_path}")
    log_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "logs")
    print(f"log dir directory is {log_path}")
    save_path = load_path

    if 'model.pt' in os.listdir(load_path):
        model_path = os.path.join(load_path, "model.pt")

    else:
        all_models = sort([int(f.split('.pt')[0]) for f in os.listdir(load_path)])
        model_path = os.path.join(load_path, f"{all_models[-1]}.pt")

    d = torch.load(model_path)
    policy_net.load_state_dict(d['policy_net'])

parser = get_args()
init_args_for_env(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True
# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)
print(args.seed)

if args.commnet:
    policy_net = CommNetMLP(args, num_inputs, train_mode=False)
elif args.random:
    policy_net = Random(args, num_inputs)

# this is what we are working with for IC3 Net predator prey.
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

load(args.load)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

evaluator = Evaluator(args, policy_net, data.init(args.env_name, args))

st_time = time.time()

# TODO: Run for miltiple episodes.
for i in range(10):
    ep, stat, all_comms = evaluator.run_episode()
    print(stat)

total_episode_time = time.time() - st_time

print("stat is: ", stat)
print("avg comm ", stat['comm_action'] / stat['num_steps'])
print("time taken per step ", total_episode_time/stat['num_steps'])

all_comms = np.array(all_comms)
num_agents = len(all_comms[0])
for i in range(num_agents):
    print(f"for agent{i} communication is: ", all_comms[:, i])
