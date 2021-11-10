import os

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

from args import get_args
from nns.probe import Probe
from utils.game_tracker import GameTracker
from utils.util_fns import *


def get_prey_location(state, num_locations):
    prey_idx = num_locations + 1
    prey_row_idx = np.where(state[:, prey_idx] == 1)
    prey_row = state[prey_row_idx]
    # Here, have the row where the prey is, so pull out location.
    prey_loc = prey_row[0, :num_locations]
    return prey_loc


def train_probe(agent_id, from_cell_state=True):
    # 1) Load a tracker, which has already been populated by running eval.
    tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'tracker.pkl')
    tracker = GameTracker.from_file(tracker_path)
    print("tracker len", len(tracker.data))

    hidden_state_idx = 0 if from_cell_state else 1
    hidden_dim = tracker.data[0][2][hidden_state_idx].detach().numpy().shape[1]

    hidden_data = np.array([data_elt[2][hidden_state_idx][agent_id].detach().numpy() for data_elt in tracker.data])

    y_data = []
    timesteps = []
    for state, obs, _, time_idx in tracker.data:
        if env_name == "spurious":
            num_locations = tracker.data[0][0].shape[0]
            y_dim = num_locations
            data_point = get_prey_location(state, num_locations)
        elif env_name == "traffic":
            pass  # TODO
        y_data.append(data_point)
        timesteps.append(time_idx)
    y_data = np.array(y_data)

    # 2) Initialize a net to predict prey location.
    num_layers = 3 if from_cell_state else 3
    probe_model = Probe(hidden_dim, y_dim, num_layers=num_layers)

    # 3) Put all the data in a dataloader
    frac = 0.75
    train_len = int(len(hidden_data) * frac)
    train_set = TensorDataset(torch.Tensor(hidden_data[:train_len]), torch.Tensor(y_data[:train_len]),
                              torch.Tensor(timesteps[:train_len]))
    test_set = TensorDataset(torch.Tensor(hidden_data[train_len:]), torch.Tensor(y_data[train_len:]),
                             torch.Tensor(timesteps[train_len:]))
    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False)

    path_name = 'c_probe' if from_cell_state else 'h_probe'
    path_name += '_' + str(agent_id) + '.pth'
    probe_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), path_name)
    if args.train:
        # 4) Do the training.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(probe_model.parameters(), lr=0.001, momentum=0.9)
        min_eval_loss = None
        max_patience = 10
        curr_patience = 0
        for epoch in range(1000):  # Was 100
            if curr_patience >= max_patience:
                print("Breaking because of patience with eval loss", min_eval_loss)
                break
            running_loss = 0.0
            for i, data in enumerate(train_dataloader):
                inputs, labels, _ = data
                optimizer.zero_grad()
                outputs = probe_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss
            # Calculate eval loss too for early stopping
            probe_model.eval()
            with torch.no_grad():
                running_loss = 0.0
                for data in test_dataloader:
                    inputs, labels, _ = data
                    outputs = probe_model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss
                print("Eval loss", running_loss)
                if min_eval_loss is None or running_loss < min_eval_loss:
                    min_eval_loss = running_loss
                    curr_patience = 0
            curr_patience += 1
            probe_model.train()
        torch.save(probe_model.state_dict(), probe_path)
    probe_model.load_state_dict(torch.load(probe_path))
    # 5) Do some eval
    correct = 0
    total = 0
    true_loc_hist = {}
    pred_loc_hist = {}
    correct_by_time = {}
    total_by_time = {}
    all_true = []
    all_pred = []
    with torch.no_grad():
        for data in test_dataloader:
            images, labels, time_idxs = data
            labels = torch.argmax(labels, dim=1)
            outputs = probe_model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_true.extend(labels)
            all_pred.extend(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label in labels:
                val = label.item()
                if val not in true_loc_hist.keys():
                    true_loc_hist[val] = 0
                true_loc_hist[val] += 1
            for pred in predicted:
                val = pred.item()
                if val not in pred_loc_hist.keys():
                    pred_loc_hist[val] = 0
                pred_loc_hist[val] += 1
            # Time-based metrics
            for idx, time_idx in enumerate(time_idxs):
                time_val = int(time_idx.item())
                if time_val not in total_by_time.keys():
                    total_by_time[time_val] = 0
                    correct_by_time[time_val] = 0
                total_by_time[time_val] += 1
                if predicted[idx] == labels[idx]:
                    correct_by_time[time_val] += 1
    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    # Plot accuracy over time in episode
    times_to_plot = [i for i in range(20)]
    accuracies = [0 for i in range(20)]
    for time_idx in times_to_plot:
        if total_by_time.get(time_idx) is not None:
            accuracies[time_idx] = correct_by_time[time_idx] / total_by_time[time_idx]
    plt.plot(times_to_plot, accuracies)
    title = "C Probe" if from_cell_state else "H Probe"
    title += " for agent " + str(agent_id)
    plt.title(title)
    plt.xlabel("Time step into episode (max 20)")
    plt.ylabel("Probe accuracy")
    plt.show()

    # Plot a confusion matrix over prey locations.
    confusion = confusion_matrix(all_true, all_pred)
    plt.matshow(confusion)
    plt.show()


if __name__ == '__main__':
    env_name = "traffic"  # FIXME: from args
    parser = get_args()
    init_args_for_env(parser)
    args = parser.parse_args()
    args.train = True  # Set by hand for now.
    for agent_id in [0, 1]:
        for do_c in [True, False]:
            train_probe(agent_id, do_c)
