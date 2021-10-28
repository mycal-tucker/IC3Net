import os
import signal
import argparse
from tensorboardX import SummaryWriter
# import visdom
import data
from nns.models import *
from comm import CommNetMLP
from utils.util_fns import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer
from collections import defaultdict

# note for adding a new env: Add it in the data.py. Might involve righting a file in ic3net_envs.

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='trained_models', type=str,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='trained_models', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')


parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')
parser.add_argument('--log_dir', default='tb_logs', type=str,
                    help='directory to save tensorboard logs')
parser.add_argument('--exp_name', default='default_exp', type=str,
                    help='directory to save tensorboard logs')

# TODO: Sanity check so as to make sure discrete and proto works for environments other than predator-prey.
#  Currently the discrete and prototype based methods will only really take effect from inside the CommNet.
parser.add_argument('--use_proto', default=False, action='store_true',
                    help='Whether to use prototype nets in the communication layer.')

parser.add_argument('--discrete_comm', default=False, action='store_true',
                    help='Whether to use discrete_comm')
parser.add_argument('--num_proto', type=int, default=6,
                    help="Number of prototypes to use")

parser.add_argument('--comm_dim', type=int, default=128,
                    help="Dimension of the communication vector")


# TODO: Formalise this gating head penalty factor
parser.add_argument('--gating_head_cost_factor', type=float, default=0.0,
                    help='discount factor')
parser.add_argument('--restore', action='store_true', default=False,
                    help='plot training progress')

# first add environment specific args to the parser
init_args_for_env(parser)

args = parser.parse_args()

if args.ic3net:
    # if using ic3 net commnet and hard_attn both are on.
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    # if args.env_name == "traffic_junction":
    #     # args.comm_action_one = True
    #     args.comm_action_one = False

# Enemy comm
# this is set to 3 in predator-prey -> for the basic version. You can compare for other versions too.
args.nfriendly = args.nagents

# for the default instruction there is no enemy communication.
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)
print("env action space is: ", env.action_space)

num_inputs = env.observation_dim
# this basically is 4 for the case of basic predator-prey
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs
print("num inputs is: ", num_inputs)
# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    # so communication has been made part of the action now
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1


# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'

# this basically introduces the multi-head action strategy.
parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)

# since for ic3net case, commnet is true policy is commnetMLP.
if args.commnet:
    print("using commnet")
    # TODO: You can try moving this to device
    policy_net = CommNetMLP(args, num_inputs)
elif args.random:
    print("Using random")
    policy_net = Random(args, num_inputs)
# this is what we are working with for IC3 Net predator prey.
elif args.recurrent:
    print("Using rnn")
    policy_net = RNN(args, num_inputs)
else:
    print("Using policynet")
    policy_net = MLP(args, num_inputs)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.nprocesses > 1:
    # this is the main trainer. This is where the environment is being passed.
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))

else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True

def disp():
    x = disp_trainer.get_episode()

# definition of LogField
# LogField = namedtuple('LogField', ('data', 'plot', 'x_axis', 'divide_by'))

# old visdom code probably not needed anymore
log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')

# define save dirctory
# logs will also be saved under the same directory
# TODO: For loading similar arrangements need to be made.
if not args.restore:
    save_path = os.path.join(args.save, args.env_name, args.exp_name, "seed" + str(args.seed), "models")
    print(f"save directory is {save_path}")
    log_path = os.path.join(args.save, args.env_name, args.exp_name, "seed" + str(args.seed), "logs")
    print(f"log dir directory is {log_path}")
    os.makedirs(save_path, exist_ok=True)

    # if os.path.exists(log_path):
    #     shutil.rmtree(log_path)
    #     print("delete log done")
    assert os.path.exists(log_path) == False, "The save directory already exists, use load instead if you want to continue" \
                                              " training" + str(log_path)
    os.makedirs(log_path, exist_ok=True)
    logger = SummaryWriter(log_path)

# Removed this as we dont need to check if we want to plot or not. We plot all the time.
# if args.plot:
    # vis = visdom.Visdom(env=args.plot_env)
    # logger = SummaryWriter(str(os.path.join(args.env_name, args.log_dir)))


# this is used for getting that multiple seed plot in the end.
history = defaultdict(list)
start_epoch  = 0

def run(num_epochs):
    # for ep in range(start_epoch, start_epoch + num_epochs):
    for ep in range(start_epoch, num_epochs):
        epoch_begin_time = time.time()
        stat = dict()

        # added to store stats to numpy array

        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            # print(f"train batch called")
            s = trainer.train_batch(ep)
            # print(f"train batch completed")
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        # epoch = len(log['epoch'].data) + 1
        epoch = ep
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        for k, v in stat.items():
            if k == "comm_action" or k == "reward":
                for i, val in enumerate(v):
                    logger.add_scalar(f"agent{i}/{k}" , val, epoch)
                    history[f"agent{i}_{k}"].append(val)

            elif k != "epoch":
                logger.add_scalar(k, v, epoch)
                history[k].append(v)

        # print(stat)
        np.set_printoptions(precision=2)

        print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
                epoch, stat['reward'], epoch_time
        ))

        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.2f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))

        # old visdom code probably not needed anymore.

        # if args.plot:
        #     for k, v in log.items():
        #         if v.plot and len(v.data) > 0:
        #             vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
        #             win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        if args.save_every and ep and args.save != '' and ep % args.save_every == 0:
            # fname, ext = args.save.split('.')
            # save(fname + '_' + str(ep) + '.' + ext)
            # TODO: Add seed to this path as well.
            # save(args.save + '_' + str(ep))
            save(save_path + '/' + str(ep) +'.pt')

            # also save history periodically.
            for k, v in history.items():
                value = np.array(v)
                np.save(f"{log_path}/{k}.npy", value)

        if args.save != '':
            # save(args.save)
            save(save_path + '/model.pt')


        # save history to the relevant path
        for k, v in history.items():
            value = np.array(v)
            np.save(f"{log_path}/{k}.npy", value)

def save(path):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    torch.save(d, path)

def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])

    # TODO: this need to be loading the tensorboard logs instead.
    # log.update(d['log'])   # this was for the visdom logs.

    # load history as well to continue training.
    # for f in os.listdir(log_path):
    #     if f.endswith(".npy"):
    #         k = f.split('.npy')[0]
    #         history[k] = list(np.load(f"{log_path}/{k}.npy"))

    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# TODO: load needs to be changed similar to save
# if args.load != '':
#     load(args.load)

if args.restore:
    load_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "models")
    print(f"load directory is {load_path}")
    log_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "logs")
    print(f"log dir directory is {log_path}")
    logger = SummaryWriter(log_path)

    save_path = load_path

    if 'model.pt' in os.listdir(load_path):
        model_path = os.path.join(load_path, "model.pt")

    else:
        all_models = sort([int(f.split('.pt')[0]) for f in os.listdir(load_path)])
        model_path = os.path.join(load_path, f"{all_models[-1]}.pt")

    d = torch.load(model_path)
    policy_net.load_state_dict(d['policy_net'])

    # load history as well to continue training.
    for f in os.listdir(log_path):
        if f.endswith(".npy"):
            k = f.split('.npy')[0]
            history[k] = list(np.load(f"{log_path}/{k}.npy"))
            start_epoch = len(history[k])

    trainer.load_state_dict(d['trainer'])


run(args.num_epochs)

if args.display:
    env.end_display()

if args.save != '':
    # save(args.save)
    save(save_path + '/model.pt')

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
