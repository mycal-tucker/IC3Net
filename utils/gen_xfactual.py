import os
from settings import settings
import torch.nn as nn
import torch.optim as optim
import time
from args import get_args
from nns.probe import Probe
from utils.game_tracker import GameTracker
from utils.util_fns import *


def gen_counterfactual(z, probe, s_prime, criterion=None):
    start_time = time.time()
    z_prime = z
    z_prime.requires_grad = True
    # optimizer = optim.SGD([z_prime], lr=0.0001, momentum=0.9)
    # optimizer = optim.SGD([z_prime], lr=0.001, momentum=0.9)
    optimizer = optim.SGD([z_prime], lr=0.01, momentum=0.9)  # Good. Generated the prey results.
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    num_steps = 0
    stopping_loss = 0.001  # Was 0.05
    # stopping_loss = .001  # Generated the prey results
    loss = 100
    max_patience = 10000
    max_num_steps = settings.NUM_XFACT_STEPS
    curr_patience = 0
    min_loss = loss
    probe.eval()
    curr_time = time.time()
    while curr_time - start_time < 100 and num_steps < max_num_steps and loss > stopping_loss:
        optimizer.zero_grad()
        outputs = probe(torch.Tensor(z_prime))
        loss = criterion(outputs, s_prime)
        loss.backward()
        optimizer.step()
        # if num_steps % 100 == 0:
        #     print("Loss", loss)
        num_steps += 1
        curr_patience += 1
        if loss < min_loss - 0.01:
            min_loss = loss
            curr_patience = 0
        if curr_patience > max_patience:
            print("Breaking because of patience with loss", loss)
            break
        curr_time = time.time()
        # print("Diff", curr_time - start_time)
    # print("Num steps", num_steps, "\tloss", loss)
    # if num_steps >= max_num_steps:
    #     print("Max step thing", loss)
    # if loss <= stopping_loss:
    #     print("Broke for min loss", loss)
    return z_prime


if __name__ == '__main__':
    print("Dummy calls for debugging.")
    # First, load a tracker and a trained probe.
    parser = get_args()
    init_args_for_env(parser)
    args = parser.parse_args()
    tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'tracker.pkl')
    tracker = GameTracker.from_file(tracker_path)

    probe_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'c_probe.pth')

    c_dim = tracker.data[0][2][0].detach().numpy().shape[1]
    num_locations = tracker.data[0][0].shape[0]
    c_probe = Probe(c_dim, num_locations, num_layers=3)
    c_probe.load_state_dict(torch.load(probe_path))
    c_probe.eval()

    new_goal = np.zeros((1, num_locations))
    new_goal[0, 20] = 1
    new_goal = torch.Tensor(new_goal)
    agent_id = 3
    for _, _, hiddens in tracker.data:
        c, _ = hiddens
        c = c[agent_id].detach().numpy()
        c = torch.unsqueeze(torch.Tensor(c), 0)
        xfactual = gen_counterfactual(c, c_probe, new_goal)
        print("X factual", xfactual)
