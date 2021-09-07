import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-5        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_WEIGHT_DECAY = 0.7
NOISE_WEIGHT_START = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    
    def __init__(self, state_size, action_size, random_seed, noise_weight = NOISE_WEIGHT_START, 
                 noise_weight_decay = NOISE_WEIGHT_DECAY):
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
        self.noise_w = noise_weight
        self.noise_wd = noise_weight_decay

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > BATCH_SIZE and done:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
                
        self.actor_local.train()
        if add_noise:
            noise_sample = np.random.normal(scale=1) * self.noise_w
            self.noise_w = self.noise_w * self.noise_wd
            action += noise_sample
        return np.clip(action, -50, 50)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * dones)
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def train_ddpg(self, env, n_episodes = 1000, print_every = 100, stop = True):
        """Train the agent in the Forcefield environment using ddpg. 
        Params
        ======
            env: environment e.g. of type Forcefield
            n_episodes: int, max number of episodes training
            print_every: int, print new line with average scores every n episodes. 
            stop: bool, stops training if task is learned. 
        """
        trajectories = Trajectories(env)
    
        scores = []
        actions_tracker = []
        scores_deque = deque(maxlen=print_every)
        solved = False

        for i_episode in range(n_episodes):
            
            env_info = env.reset()
            state = env_info.state        # current state
            score = 0                     # initialize agent scores
            trajectory = [state[:2]]      # initialize trajectory 
            actions = [state[2:4]]

            while True:

                action = self.act(state)

                env_info = env.step(action)               # send action to environment
                next_state = env_info.state               # get next state 
                reward = env_info.reward                  # get reward 
                done = env_info.done                      # see if trial is finished

                self.step(state, action, reward, next_state, done)

                score += reward                         # update the score (for each agent)
                state = next_state                               # enter next states
                trajectory.append(env_info.pos)
                actions.append(action)

                if done:
                    break

            trajectories.add_episode(trajectory)

            scores_deque.append(np.mean(score))
            scores.append(np.mean(score))
            actions_tracker.append(actions)

            print('\rEpisode {} \tAverage Reward: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

            if i_episode % print_every == 0:
                torch.save(self.actor_local.state_dict(), 'actor_model.pth')
                torch.save(self.critic_local.state_dict(), 'critic_model.pth')
                print('\rEpisode {} \tAverage Reward: {:.2f}'.format(i_episode, np.mean(scores_deque)))

            if np.mean(scores_deque) >= 5:
                if not solved:
                    solved = True 
                    print('\nEnvironment solved in {:d} episodes!\t Average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                    torch.save(self.actor_local.state_dict(), 'actor_solved.pth')
                    torch.save(self.critic_local.state_dict(), 'critic_solved.pth')
                
            if solved and stop:
                break

        return scores, trajectories, actions_tracker
        

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
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
        """Return the current size of internal memory."""
        return len(self.memory)

    
class Trajectories():
    
    def __init__(self, env):
        """"
        Params
        ======
        trajectories = list of trajectories, where each trajectory is an array or list of (x, y) coordinates at each timestep.
        env = environment in which trajectories were performed. indicates goal box, workspace dimensions, etc. 
        """
        
        self.trajectories = []
        self.max_len = env.max_len
        
        # goal box and workspace bounds
        self.goal = env.goal
        self.bounds = env.bounds 
        
    def add_episode(self, positions):
        self.trajectories.append(positions)
        
    def plot(self, idx, legend = False, color = 'magma', scale=True, boxcol = 'r', boxalpha = 0.1):
        """Plot a select number of indices.
        Params
        ======
        tr_id = int index of trajectory to plot
        cmap = colors to map with, from cm
        scale = if True, samples from the colormap based on length of trial. 
        """
        
        if scale:
            cmap = cm.get_cmap(color, len(self.trajectories[idx])) 
        else:
            cmap = cm.get_cmap(color, self.max_len)
            
        xlims = (self.bounds[0], self.bounds[1]) 
        ylims = (self.bounds[2], self.bounds[3])
        fig, ax = plt.subplots()

        goal_patch = patches.Rectangle((self.goal[0], self.goal[2]), self.goal[1]-self.goal[0], \
                                       self.goal[3]-self.goal[2], linewidth=1, alpha=boxalpha, \
                                       edgecolor=boxcol, facecolor=boxcol)
        
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        
        for i, pt in enumerate(self.trajectories[idx]):
            # ax.plot(pt[0], pt[1], 'o', color=cmap.colors[i])
            if legend and i == 0:
                ax.plot(pt[0], pt[1], 'o', color=cmap.colors[i], label='start')
            elif legend and i == len(self.trajectories[idx])-1:
                ax.plot(pt[0], pt[1], 'o', color=cmap.colors[i], label='end')
            else:
                ax.plot(pt[0], pt[1], 'o', color=cmap.colors[i])

        ax.add_patch(goal_patch)
        
        if legend:
            ax.legend()
        
        return fig, ax

    def plot_converged(self,legend=False,boxcol='r',boxalpha=.1):
        """Plot the results of the RL algorithm. The first subplot represents the 
        total gain per episode, the second one represents the trajectory of the last episode
        =======
        Params
        =======
        legend: if True represents the first & last dots differently
        """

        goal_patches=patches.Rectangle((self.goal[0], self.goal[2]),self.goal[1]-self.goal[0], \
                                      self.goal[3]-self.goal[2],lw=1,alpha=boxalpha,edgecolor=boxcol,facecolor=boxcol)

        fig, axs=plt.subplots(1,2)

        axs[0].plot(range(len(self.scores)),self.scores,'b',lw=2)
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total reward')
        axs[0].set_title('Reward per episode')

        for i,pt in enumerate(self.trajectories[-1]):
            if legend and i==0:
                axs[1].plot(pt[0],pt[1],'g+',label='start',ms=15)
            elif legend and i==len(self.trajectories[-1])-1:
                axs[1].plot(pt[0],pt[1],'ro',label='end',ms=15)
            else:
                continue

        axs[1].plot(self.trajectories[-1][0],self.trajectories[-1][1],'k-',lw=2)
        axs[1].add_patch(goal_patches)
        axs[1].set_xlabel('x-position')
        axs[1].set_ylabel('y-position')
        axs[1].set_title('Last trajectory')
        plt.show()

        return fig,axs

    def plot_kinematics(self,idx):
        """Plot the kinematics parameters of the trajectory associated with index 'idx'
        =======
        Params
        =======
        idx is the index of the trajectory we want to investigate
        """
        xpos,ypos=[],[]

        for i,pt in enumerate(self.trajectories[idx]):
            xpos.append(pt[0])
            ypos.append(pt[1])

        maxtime=len(xpos)
        fig, axs=plt.subplots(2,2,sharex=True)
        axs[0,0].plot(range(maxtime),xpos,'r-',lw=2)
        axs[0,0].set_ylabel('x-position')

        axs[0,1].plot(range(maxtime),ypos,'r-',lw=2)
        axs[0,1].set_ylabel('y-position')
        axs[0,1].yaxis.tick_right()
        axs[0,1].yaxis.set_label_position('right')

        axs[1,0].plot(range(maxtime-1),np.diff(np.array(xpos)),'r-',lw=2)
        axs[1,0].set_ylabel('x-velocity')
        axs[1,0].set_xlabel('Time')

        axs[1,1].plot(range(maxtime-1),np.diff(np.array(ypos)),'r-',lw=2)
        axs[1,1].set_ylabel('y-velocity')
        axs[1,1].set_xlabel('Time')
        axs[1,1].yaxis.tick_right()
        axs[1,1].yaxis.set_label_position('right')
        plt.show()

        return fig, axs
    
    
