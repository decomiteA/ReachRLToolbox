import numpy as np 
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from replay_buffer import ReplayBuffer

import torch
import torch.optim as optim
import torch.functional as F


BUFFER_SIZE = int(1e5)
BATCHSIZE = 20
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-10
LR_CRITIC = 1e-10
WEIGHT_DECAY = 1e-8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ Interacts with and learns from the environment"""
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """ Initialize an agent object
        Params
        ======
            state_size (int) : dimension of each state
            action_size (int): dimension of each action 
            random_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size= action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        #Actor network 
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #Critic network 
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY)

        #Noise process

        self.noise = OUNoise((num_agents, action_size), random_seed)

        #Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """ Save experiences in replay memory, and use random sample from buffer to learn"""
        #Save experience/reward 
        for agent in range(self.num_agents):
           self.memory.add(states[agent,:], actions[agent,:], rewards, next_states[agent,:], dones)

        #Learn, if enough samples are available 
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def reset(self):
        self.noise.reset()

    def learn(self,experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + gamma*critic_target(next_state, actor_target(next_state))
        where:
            actor_target(next_state) = action
            critic_target(state,action) = Q_value

        Params
        ======
            experiences (Tuple(torch.tensor)):tuple of (s,a,r,s',done) tuples
            gamma (float): discount factor 
        """
        states, actions, rewards, next_states, dones = experiences
        # Update critic
        actions_next = self.actor_target(next_states)
        Q_target_next = self.critic_target(next_states,actions_next)
        #comupte q_target for current state (y_i)
        Q_target = rewards + (GAMMA * Q_targets_next * (1-dones))
        #compute critic loss
        Q_expected = self.critic_local(states,actions)
        critic_loss = F.mse_loss(Q_expected, Q_target)
        #minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        #update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #update target networks 
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param,local_param in zip(target_model.parameters(),local_model.parameter()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein uhlenbeck"""
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.15, sigma_min=0.05, sigma_decay=.975):
        self.mu = mu *np.ones(size)
        self.theta = theta
        self.sigma = sigma 
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = seed 
        self.size = size 
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        x = self.state
        dx  = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
        
