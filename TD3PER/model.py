import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

##### HYPERPARAMETERS #####
# Number of units in the first hidden layer
fc1_units = 1024
# Number of units in the second hidden layer
fc2_units = 512
fc3_units = 64
lstm_hidden_size = 256  # Output size of the LSTM layer

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Actor, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=state_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.state_size=state_size
        self.action_size=action_size
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        batch_size = x.size(0)
        sequence_length = x.size(1)

        
        x = x.view(batch_size ,1,sequence_length)  # Flatten sequence and batch dimensions
        lstm_out, _ = self.lstm(x)
        # Take only the last LSTM output for each sequence
        lstm_out = lstm_out[:, -1, :]

        # Pass through fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output action values
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=state_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        batch_size = x.size(0)
        sequence_length = x.size(1)
        x = x.view(batch_size ,1,sequence_length)  # Flatten sequence and batch dimensions

        # Pass through LSTM layer

        lstm_out, _ = self.lstm(x)

        # Take only the last LSTM output for each sequence
        lstm_out = lstm_out[:, -1, :]

        # Pass through fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output Q-values
        return self.fc4(x)
