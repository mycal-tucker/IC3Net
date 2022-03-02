import os
from curses import wrapper
import csv
import numpy as np
from settings import settings

from utils import data
from action_utils import parse_action_args
from args import get_args
from comm import CommNetMLP
from evaluator import Evaluator
from nns.models import *
from utils.util_fns import *
from utils.game_tracker import GameTracker

from nns.probe import Probe

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


def load_parent_child_probes(probe_seed, probe_dropout_rate):
    tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'tracker.pkl')
    old_tracker = GameTracker.from_file(tracker_path)
    c_dim = old_tracker.data[0][2][0].detach().numpy().shape[1]
    probe_pred_dim = 49
    c_probes = [Probe(c_dim, probe_pred_dim, num_layers=3) for _ in range(2)]
    [c_probe.load_state_dict(torch.load(os.path.join(args.load, args.env_name, args.exp_name, "seed" +
                                                     str(args.seed), 'probes', 'seed' + str(probe_seed),
                                                     'c_probe_' + str(i) + '_dropout_' + str(
                                                         probe_dropout_rate) + '.pth'))) for i, c_probe in
     enumerate(c_probes)]
    [c_probe.eval() for c_probe in c_probes]

    h_probes = [Probe(c_dim, probe_pred_dim, num_layers=3) for _ in range(self.num_agents)]
    [h_probe.load_state_dict(torch.load(
        os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'probes',
                     'seed' + str(probe_seed), 'h_probe_' +
                     str(i) + '_dropout_' + str(probe_dropout_rate) + '.pth'))) for i, h_probe in
     enumerate(h_probes)]
    [h_probe.eval() for h_probe in h_probes]
    return c_probes, h_probes


def run_eval(_):
    def load(path):
        load_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "models")
        print(f"load directory is {load_path}")
        log_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "logs")
        print(f"log dir directory is {log_path}")

        assert 'model.pt' in os.listdir(load_path), "No model to load!?"
        model_path = os.path.join(load_path, "model.pt")
        d = torch.load(model_path)
        policy_net.load_state_dict(d['policy_net'])

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
        args.seed = np.random.randint(0, 10000)
    torch.manual_seed(args.seed)

    print(args)
    print(args.seed)

    if args.commnet:
        print("Creating commnet mlp")
        policy_net = CommNetMLP(args, num_inputs, train_mode=False)
    elif args.random:
        policy_net = Random(args, num_inputs)

    # this is what we are working with for IC3 Net predator prey.
    elif args.recurrent:
        print("Creating an RNN!")
        policy_net = RNN(args, num_inputs)
    else:
        policy_net = MLP(args, num_inputs)

    load(args.load)

    if not args.display:
        display_models([policy_net])

    # share parameters among threads, but not gradients
    for p in policy_net.parameters():
        p.data.share_memory_()

    if args.use_tracker:
        evaluator = Evaluator(args, policy_net, data.init(args.env_name, args), 0, 0,
                              -1)
        all_stats = []
        for i in range(1000):
            ep, stat, all_comms = evaluator.run_episode()
            all_stats.append(stat)
            if i % 50 == 0:
                print("Episode number", i)
        tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "tracker.pkl")
        evaluator.tracker.to_file(tracker_path)
        return

    # num_steps = [i for i in range(1, 2)]
    # For traffic, always intervene
    num_steps = [10]
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # dropout_rates = [0.1]
    # dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    # dropout_rates = [0.0]
    probe_seeds = [i for i in range(0, 5)]
    # probe_seeds = [0]
    # xfact_steps = [50, 250, 500, 750, 1000, 2000, 3000, 4000, 5000]
    xfact_steps = [200]
    success_table = np.zeros((len(dropout_rates), len(num_steps), 2))
    collision_table = np.zeros((len(dropout_rates), len(num_steps), 2))
    time_table = np.zeros((len(dropout_rates), len(num_steps), 2))
    for xfact_step in xfact_steps:
        settings.NUM_XFACT_STEPS = xfact_step
        for inter_idx, num_intervention_steps in enumerate(num_steps):
            for dropout_idx, probe_dropout_rate in enumerate(dropout_rates):
                succ_for_seed = []
                collision_for_seed = []
                time_for_seed = []
                for probe_seed in probe_seeds:
                    print("Eval for dropout", probe_dropout_rate, "for", num_intervention_steps, "steps for probe seed", probe_seed)
                    evaluator = Evaluator(args, policy_net, data.init(args.env_name, args), probe_dropout_rate, probe_seed, num_intervention_steps)
                    st_time = time.time()
                    all_stats = []
                    for i in range(100):
                        ep, stat, all_comms = evaluator.run_episode()
                        all_stats.append(stat)
                        if i % 20 == 0:
                            print("Episode", i)
                    if args.use_tracker:
                        tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "tracker.pkl")
                        evaluator.tracker.to_file(tracker_path)

                    total_episode_time = time.time() - st_time
                    average_stat = {}
                    for key in all_stats[0].keys():
                        average_stat[key] = np.mean([stat.get(key) for stat in all_stats])
                    print("average stats is: ", average_stat)
                    succ_for_seed.append(average_stat.get('success'))
                    if average_stat.get('collisions') is not None:
                        collision_for_seed.append(average_stat.get('collisions'))
                    time_for_seed.append(total_episode_time/average_stat['num_steps'])
                success_table[dropout_idx, inter_idx, 0] = np.mean(succ_for_seed)
                success_table[dropout_idx, inter_idx, 1] = np.std(succ_for_seed)
                collision_table[dropout_idx, inter_idx, 0] = np.mean(collision_for_seed)
                collision_table[dropout_idx, inter_idx, 1] = np.std(collision_for_seed)
                time_table[dropout_idx, inter_idx, 0] = np.mean(time_for_seed)
                time_table[dropout_idx, inter_idx, 1] = np.std(time_for_seed)
                # print("Success table\n", success_table[:, :, 0])
                # print("Collision table\n", collision_table[:, :, 0])
                # print("Success std table\n", success_table[:, :, 1])
                # print("Collision std table\n", collision_table[:, :, 1])
                with open(os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed),
                                       "success_table_" + str(xfact_step) + ".csv"), 'w') as f:
                    f.write("Success mean\n")
                    writer = csv.writer(f)
                    writer.writerows(success_table[:, :, 0])
                    f.write("Success std\n")
                    writer.writerows(success_table[:, :, 1])
                    f.write("Collision mean\n")
                    writer.writerows(collision_table[:, :, 0])
                    f.write("Collision std\n")
                    writer.writerows(collision_table[:, :, 1])
                    f.write("Time mean\n")
                    writer.writerows(time_table[:, :, 0])
                    f.write("Time std\n")
                    writer.writerows(time_table[:, :, 1])


if __name__ == '__main__':
    # Wrap entire execution in curses wrapper to protect terminal against ugly end states.
    parser = get_args()
    init_args_for_env(parser)
    args = parser.parse_args()
    if args.display:
        wrapper(run_eval)
    else:
        run_eval(None)
