from tqdm import tqdm
import os
import os

import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from args import get_args
from utils.game_tracker import GameTracker
from utils.util_fns import *


class Net(nn.Module):
    def __init__(self, input1_size, input2_size):
        H = 10
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input1_size, H)
        self.fc2 = nn.Linear(input2_size, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2

def get_activation_correlations():
    # 1) Load a tracker, which has already been populated by running ev
    tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'tracker.pkl')
    tracker = GameTracker.from_file(tracker_path)
    print("tracker len", len(tracker.data))

    hidden_state_idx = 0 if do_c else 1
    # Create the training dataset from tracker, dividing h into halves.
    hidden_data = np.array([data_elt[2][hidden_state_idx][agent_id].detach().numpy() for data_elt in tracker.data])
    print("Hidden data", hidden_data.shape)
    hidden_data_dim = hidden_data.shape[1]
    cutoff = int(hidden_data_dim / 2)
    var1 = hidden_data[:, :cutoff]
    var2 = hidden_data[:, cutoff:]

    get_info(var1, var2, title="Agent " + str(agent_id) + " Mutual Information for top and bottom half of " + ("C" if do_c else "H"),
             do_plot=True)


def get_info(var1, var2, title="", do_plot=False):
    dim1 = var1.shape[1]
    dim2 = var2.shape[1]

    dataset = TensorDataset(torch.Tensor(var1), torch.Tensor(var2))
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create the MINE net
    model = Net(dim1, dim2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    tracked_info = []
    for _ in tqdm(range(50)):
        for data in train_dataloader:
            x_sample = data[0]
            y_sample = data[1]
            y_shuffle = torch.Tensor(np.random.permutation(y_sample))
            pred_xy = model(x_sample, y_sample)
            pred_x_y = model(x_sample, y_shuffle)
            ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            loss = - ret  # maximize
            tracked_info.append(ret.data.numpy())
            model.zero_grad()
            loss.backward()
            optimizer.step()
    if do_plot:
        plot_x = np.arange(len(tracked_info))
        plot_y = np.array(tracked_info).reshape(-1,)
        y_smoothed = gaussian_filter1d(plot_y, sigma=5)
        plt.plot(plot_x, y_smoothed)
        plt.title(title)
        plt.show()
    return tracked_info


if __name__ == '__main__':
    parser = get_args()
    init_args_for_env(parser)
    args = parser.parse_args()
    env_name = args.env_name
    for agent_id in range(0, 2):
        for do_c in [True, False]:
            get_activation_correlations()
