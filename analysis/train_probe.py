import torch
import torch.optim as optim
import torch.nn as nn
from utils.game_tracker import GameTracker
import numpy as np
from nns.probe import Probe
from torch.utils.data import TensorDataset, DataLoader

# Proof of concept of training a probe.
# TODO: break into reusable chunks eventually so we can train a probe inline with other scripts.

# 1) Load a tracker, which has already been populated by running eval.
tracker = GameTracker.from_file('trained_models/spurious/proto_fixed/seed1/tracker.pkl')
print("tracker len", len(tracker.data))

# Which agent are you going to use for the hidden states and observations.

agent_id = 2
# Examples
c0 = tracker.data[0][2][0].detach().numpy()
h0 = tracker.data[0][2][1].detach().numpy()

c_dim = tracker.data[0][2][0].detach().numpy().shape[1]
h_dim = tracker.data[0][2][1].detach().numpy().shape[1]

c_data = np.array([data_elt[2][0][agent_id].detach().numpy() for data_elt in tracker.data])

y_data = []
num_locations = tracker.data[0][0].shape[0]
for state, _, _ in tracker.data:
    # Find location of the prey.
    prey_idx = num_locations + 1
    prey_row_idx = np.where(state[:, prey_idx] == 1)
    prey_row = state[prey_row_idx]
    # Here, have the row where the prey is, so pull out location.
    prey_loc = prey_row[0, :num_locations]
    y_data.append(prey_loc)

y_data = np.array(y_data)

# 2) Initialize a net to predict prey location.
c_probe = Probe(c_dim, num_locations)

# 3) Put all the data in a dataloader
my_dataset = TensorDataset(torch.Tensor(c_data), torch.Tensor(y_data))
frac = 0.5
train_len = int(len(my_dataset) * frac)
test_len = len(my_dataset) - train_len
train_set, test_set = torch.utils.data.random_split(my_dataset, [train_len, test_len])
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)


# 4) Do the training.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(c_probe.parameters(), lr=0.001, momentum=0.9)
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = c_probe(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss
    print("Epoch loss", running_loss)

# 5) Do some eval
correct = 0
total = 0
true_loc_hist = {}
pred_loc_hist = {}
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        labels = torch.argmax(labels, dim=1)
        outputs = c_probe(images)
        _, predicted = torch.max(outputs.data, 1)
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
print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))

import matplotlib.pyplot as plt
plt.bar(true_loc_hist.keys(), true_loc_hist.values())
plt.bar(pred_loc_hist.keys(), pred_loc_hist.values(), alpha=0.5)
plt.show()
