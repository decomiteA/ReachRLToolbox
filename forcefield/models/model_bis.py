import numpy as np

import torch
import torch.nn as nn

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, alpha, state_size, action_size, fc1_units=16, fc2_units=8):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.input_dims = state_size
        self.fc1_dims = fc1_units
        self.fc2_dims = fc2_units
        self.n_actions = action_size


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data,-f1,f1)
        torch.nn.init.uniform_(self.fc1.bias.data,-f1,f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)


        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data,-f2,f2)
        torch.nn.init.uniform_(self.fc2.bias.data,-f2,f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)


        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data,-f3,f3)
        torch.nn.init.uniform_(self.mu.bias.data,-f3,f3)
        
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=16, fc2_units=8):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = torch.tanh(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

