import numpy as np
import torch

class OneDimReach():
    def __init__(self,start_pos=.5,goal=0,space_padding=5,max_len=100,discover=False):
        """
        Initializes the environement
        """

        self.action_size = 1
        self.max_len = max_len
        self.discover = False
        self.time=0

        #start position - goal target - workspace bounds 
        self.start_pos = start_pos
        self.goal = goal
        self.bounds = (goal-space_padding, goal+space_padding)

    def dist2target(self,pos):
        """
        Computes the distance to the target
        """
        return np.square(pos-self.goal)
    def reset(self,pos=.5):
        """
        Resets the environment to the starting position
        """
        self.pos = pos
        self.state = np.array([pos,0,0])
        self.next_state = None
        self.reward = 0
        self.done = 0
        self.time = 0
        self.target_counter = 0
        return self.state
    
    def step(self,action,cost=0,stay_time=1):
        """
        Agent acts in the environment and gets the resulting next state and reward
        """

        #add time
        self.time += 1
        dt,kv,tau = 0.01,1,0.04

        #calculate new state using newtonian dynamics 
        x_pos = self.state[0] + self.state[1]*dt
        x_vel = (1-kv*dt) * self.state[1] + self.state[2]*dt
        x_force = (1-dt/tau) * self.state[2] + dt/tau * action[0] + np.random.normal(0.,0.01)

        #update position and state
        self.pos = (x_pos)
        self.speed = (x_vel)
        self.state = np.array([x_pos,x_vel,x_force])

        #reward (never reached in this case as we have a single goal target)

        if (self.bounds[0]>=self.pos or self.bounds[1]<=self.pos):
            self.done = True
            self.reward = -10-np.linalg.norm(action)*cost
        elif self.time >= self.max_len:
            self.reward = - np.linalg.norm(action)*cost
            self.reward -= self.dist2target(self.pos)*(1-self.discover)
            self.done = True
        else:
            self.reward = -np.linalg.norm(action,2)*cost
            self.done = False
        return self.state, self.reward, self.done,{}
