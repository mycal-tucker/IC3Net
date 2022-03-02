import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.gen_xfactual import gen_counterfactual


class MLP(nn.Module):
    def __init__(self, args, num_inputs):
        super(MLP, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(num_inputs, args.hid_size)
        self.affine2 = nn.Linear(args.hid_size, args.hid_size)
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])
        self.value_head = nn.Linear(args.hid_size, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def __intervention__(self, probes, inputs, s_primes):
        cloned = inputs.clone()
        for idx, h_probe in enumerate(probes):
            if not h_probe:
                continue
            # print("Probe idx", idx)
            sub_comm = inputs[0, idx]
            start_h = sub_comm.detach().numpy()
            start_h = torch.unsqueeze(torch.Tensor(start_h), 0)
            new_goal = np.zeros((1, h_probe.out_dim))
            goal_id = s_primes[idx]
            if goal_id is None:
                continue
            # print("Intervening for goal id", goal_id)
            if isinstance(goal_id, np.ndarray):
                new_goal = torch.unsqueeze(torch.Tensor(goal_id), 0)
            else:
                new_goal[0, goal_id] = 1
                new_goal = torch.Tensor(new_goal)
            x_fact_h = gen_counterfactual(start_h, h_probe, new_goal)
            cloned[0, idx] = x_fact_h

    def forward(self, x, info={}):
        h1 = self.relu(self.affine1(x))
        if 'h_probes' in info.keys():
            self.__intervention__(info.get('h_probes'), h1, info.get('s_primes'))
        h2 = self.relu(sum([self.affine2(h1), h1]))

        v = self.value_head(h2)

        if self.continuous:
            action_mean = self.action_mean(h2)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v
        else:
            return [F.log_softmax(head(h2), dim=-1) for head in self.heads], v, h1


class Random(nn.Module):
    def __init__(self, args, num_inputs):
        super(Random, self).__init__()
        self.naction_heads = args.naction_heads

        # Just so that pytorch is happy
        self.parameter = nn.Parameter(torch.randn(3))

    def forward(self, x, info={}):

        sizes = x.size()[:-1]

        v = Variable(torch.rand(sizes + (1,)), requires_grad=True)
        out = []

        for o in self.naction_heads:
            var = Variable(torch.randn(sizes + (o, )), requires_grad=True)
            out.append(F.log_softmax(var, dim=-1))

        return out, v


class RNN(MLP):
    def __init__(self, args, num_inputs):
        super(RNN, self).__init__(args, num_inputs)
        self.nagents = self.args.nagents
        self.hid_size = self.args.hid_size
        if self.args.rnn_type == 'LSTM':
            del self.affine2
            self.lstm_unit = nn.LSTMCell(self.hid_size, self.hid_size)

    def forward(self, x, info={}):
        x, prev_hid = x
        encoded_x = self.affine1(x)

        if self.args.rnn_type == 'LSTM':
            batch_size = encoded_x.size(0)
            encoded_x = encoded_x.view(batch_size * self.nagents, self.hid_size)
            output = self.lstm_unit(encoded_x, prev_hid)
            next_hid = output[0]
            cell_state = output[1]
            ret = (next_hid.clone(), cell_state.clone())
            next_hid = next_hid.view(batch_size, self.nagents, self.hid_size)
        else:
            next_hid = F.tanh(self.affine2(prev_hid) + encoded_x)
            ret = next_hid

        v = self.value_head(next_hid)
        if self.continuous:
            action_mean = self.action_mean(next_hid)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v, ret
        else:
            return [F.log_softmax(head(next_hid), dim=-1) for head in self.heads], v, ret

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

