import curses

import gym
import numpy as np
from gym import spaces


class SpuriousPredatorPreyEnv(gym.Env):
    def __init__(self,):
        self.__version__ = "0.0.1"
        self.vision = 0

        self.OUTSIDE_CLASS = 2
        self.PREY_CLASS = 1
        self.PREDATOR_CLASS = 0
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.episode_over = False
        self.stdscr = None

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)

    def init_args(self, parser):
        env = parser.add_argument_group('Prey Predator task')
        env.add_argument('--nenemies', type=int, default=1,
                         help="Total number of preys in play")
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box")
        env.add_argument('--vision', type=int, default=2,
                         help="Vision of predator")
        env.add_argument('--moving_prey', action="store_true", default=False,
                         help="Whether prey is fixed or moving")
        env.add_argument('--no_stay', action="store_true", default=False,
                         help="Whether predators have an action to stay in place")
        parser.add_argument('--mode', default='mixed', type=str,
                        help='cooperative|competitive|mixed (default: mixed)')
        env.add_argument('--enemy_comm', action="store_true", default=False,
                         help="Whether prey can communicate.")

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'moving_prey', 'mode', 'enemy_comm']
        for key in params:
            setattr(self, key, getattr(args, key))

        self.nprey = args.nenemies
        self.npredator = args.nfriendly
        self.dims = (self.dim, self.dim)
        self.stay = not args.no_stay

        if args.moving_prey:
            raise NotImplementedError

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        self.naction = 4
        if self.stay:
            self.naction += 1

        self.action_space = spaces.MultiDiscrete([self.naction])

        self.num_padded_grid_cells = (self.dims[0] + 2 * self.vision) * (self.dims[1] + 2 * self.vision)

        self.num_grid_cells = (self.dims[0] * self.dims[1])
        self.OUTSIDE_CLASS += self.num_padded_grid_cells
        self.PREY_CLASS += self.num_padded_grid_cells
        self.PREDATOR_CLASS += self.num_padded_grid_cells

        # Setting max vocab size for 1-hot encoding. We define vocab_size as the number of possible unique states!?!
        # The state space is defined by, for each visible location, what's there.
        # There are (self.dims[0] + 2 * self.vision)**2 visible locations because you can see off the grid.
        # At each location, a location can be:
        # 1) Off the grid
        # 2) Have an integer number of predators there
        # 3) Have an integer number of prey there.
        # So, the state space is num locations x 3 x max_num_predators.
        # Observations are the observations for each visible location, which includes the unique id of the location plus
        # what's there.
        self.observation_dim = self.num_padded_grid_cells + 3
        # Observation for each agent will be, for each visible cell, the location of that cell and what's in it.
        self.observation_space = spaces.Box(low=0, high=self.npredator, shape=(2 * self.vision + 1,
                                                                               2 * self.vision + 1,
                                                                               self.observation_dim), dtype=int)
        if args.seed != -1:
            np.random.seed(args.seed)

    def step(self, action):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :

            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")
        action = np.array(action).squeeze()
        action = np.atleast_1d(action)

        for i, a in enumerate(action):
            self._take_action(i, a)

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        self.episode_over = False
        self.obs = self._get_obs()
        # print(self.obs)
        debug = {'predator_locs': self.predator_loc, 'prey_locs': self.prey_loc}
        return self.obs, self._get_reward(), self.episode_over, debug

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.reached_prey = np.zeros(self.npredator)

        # Locations
        locs = self._get_coordinates()
        self.predator_loc, self.prey_loc = locs[:self.npredator], locs[self.npredator:]

        # Reset the grid
        self.grid = np.zeros(self.num_grid_cells).reshape(self.dims)
        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)
        self.empty_bool_base_grid = self._onehot_initialization()

        # stat - like success ratio
        self.stat = dict()
        self.obs = self._get_obs()
        return self.obs

    def seed(self):
        return

    def _get_coordinates(self):
        idx = np.random.choice(np.prod(self.dims), (self.npredator + self.nprey), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _get_obs(self):
        bool_base_grid = self.get_true_state()
        # Agents only observe parts of the state.
        obs = []
        for p in self.predator_loc:
            p_obs = []
            for visible_x in range(p[0] - self.vision, p[0] + self.vision + 1):
                row_obs = []
                for visible_y in range(p[1] - self.vision, p[1] + self.vision + 1):
                    single_obs = bool_base_grid[self.__idxs_to_global__(visible_x, visible_y)]
                    row_obs.append(single_obs)
                p_obs.append(np.stack(row_obs))
            obs.append(np.stack(p_obs))

        if self.enemy_comm:
            for p in self.prey_loc:
                p_obs = []
                for visible_x in range(p[0] - self.vision, p[0] + self.vision + 1):
                    row_obs = []
                    for visible_y in range(p[1] - self.vision, p[1] + self.vision + 1):
                        single_obs = bool_base_grid[self.__idxs_to_global__(visible_x, visible_y)]
                        row_obs.append(single_obs)
                    p_obs.append(np.stack(row_obs))
                obs.append(np.stack(p_obs))

        obs = np.stack(obs)
        return obs

    def get_true_state(self):
        """Returns the true state of the world rather than observations of it."""
        # Populate a grid with the true locations of everything.
        bool_base_grid = self.empty_bool_base_grid.copy()
        for i, p in enumerate(self.predator_loc):
            bool_base_grid[self.__idxs_to_global__(p[0] + self.vision, p[1] + self.vision), self.PREDATOR_CLASS] += 1
        for i, p in enumerate(self.prey_loc):
            bool_base_grid[self.__idxs_to_global__(p[0] + self.vision, p[1] + self.vision), self.PREY_CLASS] += 1
        # Then just return that grid.
        return bool_base_grid

    def _take_action(self, idx, act):
        # prey action
        if idx >= self.npredator:
            # fixed prey
            if not self.moving_prey:
                return
            else:
                raise NotImplementedError

        if self.reached_prey[idx] == 1:
            return

        # STAY action
        if act == 5:
            return

        # UP
        if act == 0 and self.grid[max(0,
                                self.predator_loc[idx][0] + self.vision - 1),
                                self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0]-1)

        # RIGHT
        elif act == 1 and self.grid[self.predator_loc[idx][0] + self.vision,
                                min(self.dims[1] -1,
                                    self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = min(self.dims[1]-1,
                                            self.predator_loc[idx][1]+1)

        # DOWN
        elif act == 2 and self.grid[min(self.dims[0]-1,
                                    self.predator_loc[idx][0] + self.vision + 1),
                                    self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = min(self.dims[0]-1,
                                            self.predator_loc[idx][0]+1)

        # LEFT
        elif act == 3 and self.grid[self.predator_loc[idx][0] + self.vision,
                                    max(0,
                                    self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1]-1)

    def _get_reward(self):
        n = self.npredator if not self.enemy_comm else self.npredator + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)

        on_prey = np.where(np.all(self.predator_loc == self.prey_loc, axis=1))[0]
        nb_predator_on_prey = on_prey.size

        if self.mode == 'cooperative':
            reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
        elif self.mode == 'competitive':
            if nb_predator_on_prey:
                reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey
        elif self.mode == 'mixed':
            reward[on_prey] = self.PREY_REWARD
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        self.reached_prey[on_prey] = 1

        if np.all(self.reached_prey == 1) and self.mode == 'mixed':
            self.episode_over = True

        # Prey reward
        if nb_predator_on_prey == 0:
            reward[self.npredator:] = -1 * self.TIMESTEP_PENALTY
        else:
            # TODO: discuss & finalise
            reward[self.npredator:] = 0

        # Success ratio
        if self.mode != 'competitive':
            if nb_predator_on_prey == self.npredator:
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0

        return reward

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _onehot_initialization(self):
        # Each row has a unique id of the location, plus extra slots denoting:
        # 1) How many predators are there
        # 2) How many prey are there
        # 3) Whether the cell is outside or not.
        one_hot_array = np.zeros((self.num_padded_grid_cells, self.observation_dim))
        global_idx = 0
        for row_idx, row in enumerate(self.grid):
            for col_idx in range(row.shape[0]):
                one_hot_array[global_idx][global_idx] = 1
                if row_idx < self.vision or row_idx >= self.dims[0] + self.vision or\
                    col_idx < self.vision or col_idx >= self.dims[1] + self.vision:
                    one_hot_array[global_idx][-1] = 1
                global_idx += 1
        return one_hot_array

    def __idxs_to_global__(self, row, col):
        """Helper function maps a row and column to the global id; used for indexing into state."""
        return (self.dims[0] + self.vision * 2) * row + col

    def render(self, mode='human', close=False):
        grid = np.zeros(self.num_grid_cells, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'X'
            else:
                grid[p[0]][p[1]] = 'X'

        for p in self.prey_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'X' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()
