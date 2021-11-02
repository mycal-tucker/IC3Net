import os
from inspect import getargspec

from action_utils import *
from nns.probe import Probe
from utils.game_tracker import GameTracker
from utils.gen_xfactual import gen_counterfactual
from utils.util_fns import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class Evaluator:
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = args.display
        self.tracker = GameTracker(max_size=5000) if args.use_tracker else None

        # Lots of intervention-based variables for overwriting cell state, etc.
        tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'tracker.pkl')
        old_tracker = GameTracker.from_file(tracker_path)

        c_probe_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'c_probe.pth')
        h_probe_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'h_probe.pth')
        c_dim = old_tracker.data[0][2][0].detach().numpy().shape[1]
        num_locations = old_tracker.data[0][0].shape[0]
        self.c_probe = Probe(c_dim, num_locations, num_layers=3)
        self.c_probe.load_state_dict(torch.load(c_probe_path))
        self.c_probe.eval()
        self.h_probe = Probe(c_dim, num_locations, num_layers=3)
        self.h_probe.load_state_dict(torch.load(h_probe_path))
        self.h_probe.eval()

        new_goal = np.zeros((1, num_locations))
        goal_id = 0
        new_goal[0, goal_id] = 1
        self.new_goal = torch.Tensor(new_goal)
        self.intervention_agent_id = 3

    def run_episode(self, epoch=1):
        all_comms = []
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()

        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                # Perform an intervention on the cell state of the prey agent.
                if t >= 0:  # We seem to need to intervene at every timestep
                    start_c = prev_hid[0][3, :]
                    start_c = start_c.detach().numpy()
                    start_c = torch.unsqueeze(torch.Tensor(start_c), 0)
                    x_fact_c = gen_counterfactual(start_c, self.c_probe, self.new_goal)
                    old_c = prev_hid[0]
                    with torch.no_grad():
                        old_c[self.intervention_agent_id, :] = x_fact_c

                    # Same for h state
                    start_h = prev_hid[1][3, :]
                    start_h = start_h.detach().numpy()
                    start_h = torch.unsqueeze(torch.Tensor(start_h), 0)
                    x_fact_h = gen_counterfactual(start_h, self.h_probe, self.new_goal)
                    old_h = prev_hid[1]
                    with torch.no_grad():
                        old_h[self.intervention_agent_id, :] = x_fact_h

                    prev_hid = (old_c, old_h)
                x = [state, prev_hid]
                action_out, value, prev_hid = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            if self.tracker:
                full_state = self.env.get_true_state()
                self.tracker.add_data(full_state, next_state, prev_hid)
            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                # print("before ", stat.get('comm_action', 0), info['comm_action'][:self.args.nfriendly])
                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                all_comms.append(info['comm_action'][:self.args.nfriendly])
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            # if self.display and done:
            if self.display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return episode, stat, all_comms
