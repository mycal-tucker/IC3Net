import os
from inspect import getargspec

from action_utils import *
from nns.probe import Probe
from utils.game_tracker import GameTracker
from utils.util_fns import *
from train_probe import is_car_in_front, get_prey_location, get_grid_obstacles, car_occupancy_grid

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class Evaluator:
    def __init__(self, args, policy_net, env, probe_dropout_rate, probe_seed, num_intervention_steps):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = args.display
        self.tracker = GameTracker(max_size=10000) if args.use_tracker else None

        # Lots of intervention-based variables for overwriting cell state, etc.
        self.intervene = True
        self.num_intervention_steps = num_intervention_steps
        self.num_agents = args.nagents
        # self.intervene_ids = [i for i in range(self.num_agents)]
        self.intervene_ids = [0, 1]
        is_pred_prey = True
        try:
            if self.intervene:
                tracker_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), 'tracker.pkl')
                old_tracker = GameTracker.from_file(tracker_path)
                c_dim = old_tracker.data[0][2][0].detach().numpy().shape[1]
                num_layers = 3
                if is_pred_prey:
                    self.c_probes = [Probe(c_dim, dim, num_layers=num_layers) for dim in [9, 49]]
                    [c_probe.load_state_dict(torch.load(os.path.join(args.load, args.env_name, args.exp_name, "seed" +
                                                                     str(args.seed), 'adam_probes', 'seed' + str(probe_seed),
                                                                     'c_probe_' + str(i) + '_dropout_' + str(probe_dropout_rate) + '.pth'))) for i, c_probe in enumerate(self.c_probes)]


                    # self.h_probes = [Probe(c_dim, probe_pred_dim, num_layers=3) for _ in range(self.num_agents)]
                    self.h_probes = [Probe(c_dim, dim, num_layers=num_layers) for dim in [9, 49]]
                    [h_probe.load_state_dict(torch.load(os.path.join(args.load, args.env_name, args.exp_name,
                                                                     "seed" + str(args.seed), 'adam_probes',
                                                                     'seed' + str(probe_seed), 'h_probe_' +
                                                                     str(i) + '_dropout_' + str(probe_dropout_rate) + '.pth'))) for i, h_probe in enumerate(self.h_probes)]
                else:
                    easy = True
                    probe_pred_dim = 2 if easy else 9
                    self.c_probes = [Probe(c_dim, probe_pred_dim, num_layers=num_layers) for _ in range(self.num_agents)]

                    [c_probe.load_state_dict(torch.load(os.path.join(args.load, args.env_name, args.exp_name, "seed" +
                                                                     str(args.seed), 'adam_probes',
                                                                     'seed' + str(probe_seed),
                                                                     'c_probe_0_dropout_' + str(
                                                                         probe_dropout_rate) + '.pth'))) for i, c_probe
                     in enumerate(self.c_probes)]
                    if easy:
                        self.h_probes = self.c_probes
                    else:
                        self.h_probes = [Probe(c_dim, probe_pred_dim, num_layers=num_layers) for _ in range(self.num_agents)]

                        [h_probe.load_state_dict(torch.load(os.path.join(args.load, args.env_name, args.exp_name, "seed" +
                                                                         str(args.seed), 'adam_probes',
                                                                         'seed' + str(probe_seed),
                                                                         'h_probe_0_dropout_' + str(
                                                                             probe_dropout_rate) + '.pth'))) for i, h_probe
                         in enumerate(self.h_probes)]
                [c_probe.eval() for c_probe in self.c_probes]
                [h_probe.eval() for h_probe in self.h_probes]
        except AssertionError:
            print("No old tracker there, so not doing interventions.")
            self.intervene = False

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

        is_pred_prey = True

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        prev_obs = None
        for t in range(self.args.max_steps):
            did_intervene = False
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                x = [state, prev_hid]
                # Interventions are done within the commnet by setting info{} variables.
                if t <= self.num_intervention_steps and self.intervene:
                    true_state = self.env.get_true_state()
                    obs = self.env.get_obs()
                    # For the predator-prey env.
                    # num_locations = 25
                    # prey_idx = num_locations + 1
                    # prey_row_idx = np.where(true_state[:, prey_idx] == 1)
                    # prey_row = true_state[prey_row_idx]
                    # # Here, have the row where the prey is, so pull out location.
                    # prey_loc = prey_row[0, :num_locations]
                    # inserted_info = np.argmax(prey_loc)

                    info['h_probes'] = [None for _ in range(self.num_agents)]
                    info['c_probes'] = [None for _ in range(self.num_agents)]
                    info['s_primes'] = [None for _ in range(self.num_agents)]
                    for intervene_id in self.intervene_ids:
                        # For predator-prey
                        if is_pred_prey:
                            if intervene_id == 0:
                                # Predator sees obstacles
                                s_prime = get_grid_obstacles(true_state, 49)
                                if np.sum(s_prime) == 0:
                                    continue
                                if s_prime[4] == 1:
                                    # print("On obs")
                                    continue
                                info['h_probes'][intervene_id] = self.h_probes[intervene_id]
                                # info['c_probes'][intervene_id] = self.c_probes[intervene_id]
                                s_prime = get_grid_obstacles(true_state, 49)
                            elif intervene_id == 1:
                                # Prey learns where it is
                                info['h_probes'][intervene_id] = self.h_probes[intervene_id]
                                info['c_probes'][intervene_id] = self.c_probes[intervene_id]
                                s_prime = get_prey_location(true_state, 49)
                                s_prime = np.argmax(s_prime)
                        # For the traffic env.
                        else:
                            true_state = self.env.get_true_state()
                            obs = self.env.get_obs()
                            s_prime, alive = car_occupancy_grid(true_state, np.hstack([np.reshape(elt, (1, -1)) for elt in obs[intervene_id]])[0],
                                                         self.env.env)
                            if not alive:
                                continue
                            # if not alive or np.sum(s_prime) == 1:
                            #     continue
                            info['h_probes'][intervene_id] = self.h_probes[intervene_id]
                            info['c_probes'][intervene_id] = self.c_probes[intervene_id]
                            # print("I sure hope you want medium traffic recurrent interventions.")
                        did_intervene = True
                        info['s_primes'][intervene_id] = s_prime

                action_out, value, prev_hid = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                true_state = self.env.get_true_state()
                obs = self.env.get_obs()
                x = state
                if self.intervene:
                    info['h_probes'] = [None for _ in range(self.num_agents)]
                    info['c_probes'] = [None for _ in range(self.num_agents)]
                    info['s_primes'] = [None for _ in range(self.num_agents)]
                    for intervene_id in self.intervene_ids:
                        s_prime, _ = is_car_in_front(true_state,
                                                     np.hstack([np.reshape(elt, (1, -1)) for elt in obs[intervene_id]])[0],
                                                     self.env.env)
                        info['h_probes'][intervene_id] = self.h_probes[intervene_id]
                        # info['c_probes'][intervene_id] = self.c_probes[intervene_id]
                        # if s_prime != 1:  # Only intervene if we want to brake.
                        #     continue
                        info['s_primes'][intervene_id] = s_prime
                action_out, value, prev_hid = self.policy_net(x, info)
                prev_hid = prev_hid.detach()

            action = select_action(self.args, action_out, eval_mode=True, print_probs=False)
            action, actual = translate_action(self.args, self.env, action)
            full_state = self.env.get_true_state()
            # if did_intervene:
            #     print("Intervened")
            next_state, reward, done, info = self.env.step(actual)

            if self.tracker and prev_obs is not None:
                # Ignore gating action
                self.tracker.add_data(full_state, prev_obs, prev_hid, self.env.get_timestep(), actual[0])
            prev_obs = next_state
            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                info['comm_action'] = info['comm_action'] if not self.args.comm_action_zero else np.zeros(
                    self.args.nagents, dtype=int)

                # print("before ", stat.get('comm_action', 0), info['comm_action'][:self.args.nfriendly])
                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                all_comms.append(info['comm_action'][:self.args.nfriendly])
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm'] = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]

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
