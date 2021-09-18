import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=16, fc2_units=8,alpha):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            alpha (float) : learning rate 
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data,-f1,f1)
        torch.nn.init.uniform_(self.fc1.bias.data,-f1,f1)
        self.bn1 = nn.LayerNorm(fc1_units)

        self.fc2 = nn.Linear(fc1_units, fc2_units)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data,-f2,f2)
        torch.nn.init.uniform_(self.fc2.bias.data,-f2,f2)
        self.bn2 = nn.LayerNorm(fc2_units)

        f3 = 0.003
        self.mu = nn.Linear(fc2_units,action_size)
        torch.nn.init.uniform_(self.mu.weight.data,-f3,f3)
        torch.nn.init.uniform_(self.mu.bias.data,-f3,f3)
        
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc1_units=16, fc2_units=8, beta):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            beta (float) : learning rate for the critic
        """
        super(Critic, self).__init__()
        


        self.fc1 = nn.Linear(state_size, fc1_units)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data,-f1,f1)
        torch.nn.init.uniform_(self.fc1.bias.data,-f1,f1)
        self.bn1 = nn.LayerNorm(fc1_units)

        self.fc2 = nn.Linear(fc1_units, fc2_units)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data,-f2,f2)
        torch.nn.init.uniform_(self.fc2.bias.data,-f2,f2)
        self.bn2 = nn.LayerNorm(fc2_units)

        self.action_value = nn.Linear(action_size,fc2_units)
        f3=0.003
        self.q = nn.Linear(fc2_units,1)
        torch.nn.init.uniform_(self.q.weight.data,-f3,f3)
        torch.nn.init.uniform_(self.q.bias.data,-f3,f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value,action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value

