import os
from args import get_args
from utils.util_fns import *
import torch.optim as optim
import torch.nn as nn
from utils.game_tracker import GameTracker
from nns.probe import Probe
from torch.utils.data import TensorDataset, DataLoader


def train_probe(agent1, agent2):
    agent1_id, agent1_cell = agent1
    agent1_hidden_id = 0 if agent1_cell else 1
    agent2_id, agent2_cell = agent2
    agent2_hidden_id = 0 if agent2_cell else 1

    # c_dim = tracker.data[0][2][0].detach().numpy().shape[1]
    # h_dim = tracker.data[0][2][1].detach().numpy().shape[1]

    a1_hidden_data = np.array([data_elt[2][agent1_hidden_id][agent1_id].detach().numpy() for data_elt in tracker.data])
    a2_hidden_data = np.array([data_elt[2][agent2_hidden_id][agent2_id].detach().numpy() for data_elt in tracker.data])

    data_idx = -1
    data_idxs = []
    x_data = []
    y_data = []
    timesteps = []
    for state, obs, _, time_idx, action in tracker.data:
        data_idx += 1
        x_data.append(a1_hidden_data[data_idx])
        y_data.append(a2_hidden_data[data_idx])
        timesteps.append(time_idx)
        data_idxs.append(data_idx)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Initialize a net to predict prey location.
    num_layers = 3
    x_dim = a1_hidden_data.shape[1]
    y_dim = a2_hidden_data.shape[1]
    probe_model = Probe(x_dim, y_dim, num_layers=num_layers,
                        dropout_rate=dropout_rate)

    # 3) Put all the data in a dataloader
    frac = 0.75
    train_len = int(len(x_data) * frac)
    train_set = TensorDataset(torch.Tensor(x_data).to('cuda'), torch.Tensor(y_data).to('cuda'),
                              torch.Tensor(timesteps).to('cuda'))
    test_set = TensorDataset(torch.Tensor(x_data[train_len:]).to('cuda'), torch.Tensor(y_data[train_len:]).to('cuda'),
                             torch.Tensor(timesteps[train_len:]).to('cuda'))
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)

    root_dir = 'probes/seed' + str(probe_seed) + '/'
    dropout_str = f"{dropout_rate:.1f}"
    path_root = root_dir + ('c_probe' if agent1_cell else 'h_probe') + '_' + str(agent1_id)
    path_root += '_' + ('c_probe' if agent2_cell else 'h_probe') + '_' + str(agent2_id)
    path_root += '_dropout_' + dropout_str
    model_name = path_root + '.pth'
    probe_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), model_name)

    # Now do the training.
    if args.train:
        # 4) Do the training.
        probe_model.to('cuda')
        criterion = nn.MSELoss()
        optimizer = optim.SGD(probe_model.parameters(), lr=0.001, momentum=0.9)
        min_eval_loss = None
        max_patience = 10
        curr_patience = 0
        best_probe = Probe(x_dim, y_dim, num_layers=num_layers)
        for epoch in range(100):  # Was 100
            probe_model.train()
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
                if min_eval_loss is None or running_loss < min_eval_loss - 0.01:
                    min_eval_loss = running_loss
                    curr_patience = 0
                    best_probe.load_state_dict(probe_model.state_dict())
            curr_patience += 1
            probe_model.train()
        torch.save(best_probe.state_dict(), probe_path)
    probe_model.load_state_dict(torch.load(probe_path))


if __name__ == '__main__':
    parser = get_args()
    init_args_for_env(parser)
    args = parser.parse_args()
    env_name = args.env_name
    # Load a tracker, which has already been populated by running eval.
    tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'tracker.pkl')
    tracker = GameTracker.from_file(tracker_path)
    args.train = True  # Set by hand for now.
    for probe_seed in range(1):
        for a_id1 in range(3):
            for a1_cell in [True, False]:
                for a_id2 in range(3):
                    for a2_cell in [True, False]:
                        if a_id1 == a_id2:
                            continue

                        for dropout_rate in [0.0]:
                            print("Training probe from", a_id1, "to", a_id2)
                            train_probe((a_id1, a1_cell), (a_id2, a2_cell))
