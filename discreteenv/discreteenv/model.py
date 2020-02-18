import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class RQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, hidden_size=64):
        super(RQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.rnn = nn.LSTM(fc1_units, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.init_hidden_layer()

    def init_hidden_layer(self):
        self.hc = (0.001 * torch.randn(1, 1, self.rnn.hidden_size), 0.001 * torch.randn(1, 1, self.rnn.hidden_size))  #there are two different hidden unit
        return self.hc
        # first: time step; second: batch size; third: hidden_size

    def forward(self, state, hidden=None, update_hidden=True):
        """Build a network that maps state -> action values."""
        if hidden is None:
            hidden = self.hc
        x = F.relu(self.fc1(state))
        x = x.view(x.shape[0], -1, x.shape[1])
        x, new_hidden = self.rnn(x, hidden)
        if update_hidden:
            assert state.shape[0] == 1, "only work for one time step"
            self.hc = new_hidden
        return self.fc3(x)
