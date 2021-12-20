import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description='Plotter')
parser.add_argument('--env_name', default="predator_prey",
                    help='name of the environment to run')
parser.add_argument('--exp_name', default="default_exp",
                    help='name of the environment to run')
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--save', default='trained_models', type=str,
                    help='save the model after training')

args = parser.parse_args()
history_path = os.path.join(args.save, args.env_name, args.exp_name)  # args.seed, "models")

def plot(hist_path, num_agents):
    keys = ["entropy", "num_episodes", "num_steps", "step_taken", "success", "value_loss", 'collisions']
    for i in range(num_agents):
        keys.append(f"agent{i}_comm_action")
        keys.append(f"agent{i}_reward")

    keys = set(keys)
    history = defaultdict(list)

    min_steps = np.inf

    for seed_dir in os.listdir(hist_path):
        if seed_dir.startswith('seed'):
            for f in os.listdir(f"{hist_path}/{seed_dir}/logs"):
                if f.endswith('.npy') and f.split('.npy')[0] in keys:
                    val = np.load(f"{hist_path}/{seed_dir}/logs/{f}")
                    if len(val) < min_steps:
                    	min_steps = len(val)
                    history[f.split('.npy')[0]].append(val)


    plot_path = f"{hist_path}/graphs"
    os.makedirs(plot_path, exist_ok=True)
    for k, v in history.items():
        fig, ax = plt.subplots()
        fig.suptitle(k)
        v = np.array([t[:min_steps] for t in v])
        print(v)
        # print(f"min steps is {min_steps}")
        # print(f"{k} shape of v {v.shape}")
        v = v[:, :min_steps]
        # print(f"key is {k}, v shape is {v.shape}")
        # print(v[0].shape, v[1].shape, v[2].shape)
        mean = np.mean(v, axis=0)
        minimum = np.min(v, axis=0)
        maximum = np.max(v, axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean)
        ax.fill_between(x, minimum, maximum, alpha=0.2)
        fig.savefig(f"{plot_path}/{k}.png")

plot(history_path, args.nagents)
