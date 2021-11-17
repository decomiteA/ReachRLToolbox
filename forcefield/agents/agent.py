import numpy as np
import random
import copy
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from collections import namedtuple, deque
sys.path.append('../')

from models.model_test import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 20        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.000025         # learning rate of the actor 
LR_CRITIC = 0.00025        # learning rate of the critic
WEIGHT_DECAY = 1e-8        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size,random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        #Actor network
        self.actor_local = Actor(state_size,action_size,random_seed).to(device)
        self.actor_target= Actor(state_size,action_size,random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #Critic network
        self.critic_local = Critic(state_size,action_size,random_seed).to(device)
        self.critic_target = Critic(state_size,action_size,random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),lr=LR_CRITIC,weight_decay=WEIGHT_DECAY)

        #Noise process
        self.noise = OUNoise(action_size,random_seed)

        #Replay memory
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,random_seed)

    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        #Save experience/reward
        self.memory.add(states,actions,rewards,next_states,dones)
        #Learn, if enough samples are available
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self,state,add_noise=True):
        """Returns actions for given state as per current policy"""
        state = torch.from_numpy(state).float().to(device)
        acts = np.zeros(self.action_size)
        self.actor_local.eval()
        with torch.no_grad():
            acts = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            acts += self.noise.sample()
            return np.clip(acts,-1,1)

    def reset(self):
        self.noise.reset()

    def learn(self,experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + gamma*critic_target(next_state,actor_target(next_state))
        where: 
            actor_target(next_state) = action
            critic_target(state,action) = Q_value

        Params
        ======
            experiences (Tuple(torch.tensor)):tuple of (s,a,r,s',done) tuples
            gamma (float) : discount factor
        """
        states,actions,rewards,next_states,dones = experiences
        #Update critic
        actions_next = self.actor_target(next_states)
        Q_target_next = self.critic_target(next_states,actions_next)
        #Compute q_target for current state (y_i)
        Q_target = rewards + (GAMMA * Q_target_next * (1-dones))
        #Compute critic loss
        Q_expected = self.critic_local(states,actions)
        critic_loss = F.mse_loss(Q_expected, Q_target)
        #Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #uIdate actor 
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states,actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #Update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param,local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein Uhlenbeck noise"""
    def __init__(self,size,seed,mu=0.0,theta=0.15,sigma=0.15,sigma_min=0.05,sigma_decay=0.975):
        self.mu = mu*np.ones(size)
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
        dx = self.theta * (self.mu-x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x+dx
        return self.state    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size,buffer_size,batch_size,seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Returns the current size of internal memory"""
        return len(self.memory)

class Trajectories():

    def __init__(self,env):
        """
        Params
        ======
        Trajectories = list of trajectoreis, where each trajectory is an array or list of the state
        env = environment in which trajectories where performed
        """

        self.trajectories = []
        self.scores = []
        self.max_len = env.max_len

        #goal box and workspace bounds 
        self.goal = env.goal
        self.bounds = env.bounds

    def add_episode(self,position,scores):
        self.trajectories.append(positions)
        self.scores.append(score)

    def plot(self,idx,legend=False,color='magma',scale=True,boxcol='r',boxalpha=0.1):
        """
        Plot a selected number of indices
        Params
        ======
        idx = in index of trajectories to plot
        cmap = colors to map with, from cm
        scale = if True, samples from the colormap based on length of trial
        """

        if scale:
            cmap = cm.get_cmap(color,len(self.trajectories[idx]))
        else:
            cmap = cm.get_cmap(color,self.max_len)

        xlims = (self.bounds[0],self.bounds[1])
        fix, ax = plt.subplots()

