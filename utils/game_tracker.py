import pickle


"""Data storage class for tracking, over time, the true game state, the observations of each agent, and each agent's
hidden and cell states."""
class GameTracker:
    def __init__(self, max_size):
        self.max_size = max_size
        self.curr_idx = 0
        self.data = []

    def add_data(self, state, observations, hidden_states, timestep, actions):
        data_tuple = (state, observations, hidden_states, timestep, actions)
        if len(self.data) < self.max_size:
            self.data.append(data_tuple)
        else:
            self.data[self.curr_idx] = data_tuple
        self.curr_idx += 1
        if self.curr_idx == self.max_size:
            self.curr_idx = 0  # Roll over

    def to_file(self, filename):
        with open(filename, 'wb') as outp:
            pickle.dump(self, outp)

    @staticmethod
    def from_file(filename):
        with open(filename, 'rb') as inp:
            tracker = pickle.load(inp)
        return tracker
