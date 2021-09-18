import numpy as np
import random
import copy
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from collections import namedtuple, deque
sys.path.append('../')

from models.model import Actor, Critic

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
NOISE_WEIGHT_DECAY = 0.99
NOISE_WEIGHT_START = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    
    def __init__(self, state_size, action_size,layer1_size=100,layer2_size=100):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layerX_size (int): number of units in the hidden layer 
        """
        self.gamma = GAMMA
        self.tau = TAU
        self.memory = ReplayBuffer(action_size,state_size,BUFFER_SIZE)
        self.batch_size = BATCH_SIZE

        self.actor = Actor(state_size,action_size,layer1_size,layer2_size,LR_ACTOR)
        self.critic = Critic(state_size,action_size,layer1_size,layer2_size,LR_CRITIC)

        self.target_actor = ActorNetwork(state_size,action_size,layer1_size,layer2_size,LR_ACTOR)
        self.target_critic = CriticNetwork(state_size,action_size,layer1_size,layer2_size,LR_CRITIC)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        self.actor.eval()
        observation = torch.tensor(state,dtype=torch.float).to(self.actor.device)

        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = 10*mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)

        self.actor.train()
        return mu_prime.cpu().detach().numpy()

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
            actions = np.array([state[2],state[3]])

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
                actions=np.append(actions,action)

                if done:
                    break

            trajectories.add_episode(trajectory,score)

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

    def __init__(self, action_size,input_shape,max_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size,input_shape))
        self.new_state_memory = np.zeros((self.mem_size,input_size))
        self.action_memory = np.zeros((self.mem_size, action_size))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1-done
        self.mem_cntr +=1
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        max_mem = min(self.mem_cntr,self.mem_size)

        batch = np.random.choice(max_mem,batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    
class Trajectories():
    
    def __init__(self, env):
        """"
        Params
        ======
        trajectories = list of trajectories, where each trajectory is an array or list of (x, y) coordinates at each timestep.
        env = environment in which trajectories were performed. indicates goal box, workspace dimensions, etc. 
        """
        
        self.trajectories = []
        self.scores = []
        self.max_len = env.max_len
        
        # goal box and workspace bounds
        self.goal = env.goal
        self.bounds = env.bounds 
        
    def add_episode(self, positions,score):
        self.trajectories.append(positions)
        self.scores.append(score)
        
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
                axs[1].plot(pt[0],pt[1],'o')

        
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
    
    
