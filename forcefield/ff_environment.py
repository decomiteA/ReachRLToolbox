import numpy as np
import torch

class TargetReach():
    
    def __init__(self, start_pos = (.5, 1), goal = (.1, .9, -.8, 0), space_padding = 5, max_len=60, discover=False):
        """Initialize forcefield environment.
        Params
        ======
        start_pos: tuple (x, y) coordinates of starting position.
        goal: tuple(x1, x2, y1, y2) coordinates of goal box. note x1 < x2 and y1 < y2.
        space_padding: float space from start_pos in each direction delineating allowed movement.
        max_len: max number of timesteps per episode. 
        discover: bool, indicates whether the agent can "see" the target box or whether it needs to be discovered. 
        """
        
        assert goal[0] < goal[1] and goal[2] < goal[3], \
        "goal arg must satisfy goal[0] < goal[1] and goal[2] < goal[3]. Goal is (x1, x2, y1, y2) \
        tuple with (x1, y1) the top left corner and (x2, y2) the bottom right corner of the goal box."
        
        self.action_size = 2 # (x_acceleration, y_acceleration)
        self.max_len = max_len
        self.discover = False 
        
        # start position, goal box, workspace bounds
        self.start_pos = start_pos 
        self.goal = goal
        self.bounds = (goal[0] - space_padding, goal[1] + space_padding, goal[2] - space_padding, goal[3] + space_padding)

        # forcefields
        self.ff_force = (0, 0)
        self.ff_lim = self.bounds[-2:] # y limits only 
        
    def add_forcefield(self, force, top=None, bottom=None):
        """Add a forcefield. Default is to apply forcefield from a limit in the y dimension.
        Params
        ======
        force: (x, y) tuple of forces applied in each direction. 
        top: upper limit in y-axis of applied forcefield
        bottom: lower limit in y-axis of applied forcefield
        """
        self.ff_force = force
        if top is not None:
            self.ff_lim[0] = top
        if bottom is not None:
            self.ff_lim[1] = bottom
        
    def reset(self, pos=(.5, 1)):
        """Reset the environment to the starting position. The start position is (.5, 1) on a 2D coordinate system). 
        Params
        ======
        pos: 2-tuple, (x, y) starting coordinates of agent in workspace. 
        """
        
        self.pos = pos
        self.state = np.array(list(pos) + [0, 0, 0, 0]) # State is [x_position, y_position, x_velocity, y_velocity].
        self.next_state = None
        self.reward = 0
        self.done = 0
        self.time = 0
        self.target_counter = 0
        
        return self 
    
    def dist2target(self, pos):
        """Calculates the euclidian/shortest distance of agent from target box. Used to calculate the error cost applied to \
        reward when discovery=False, i.e. the agent can "see" how far it is from the target box. 
        """
        dist = np.sqrt(min(np.absolute(pos[0]-self.goal[0]),np.absolute(pos[0]-self.goal[1]))**2+min(np.absolute(pos[1]-self.goal[2]),np.absolute(pos[1]-self.goal[3]))**2)
        return dist
    
        
    def step(self, action, cost = 0.002, stay_time=1):
        """Agent acts in the environment and gets the resulting next state and reward obtained.
        The system dynamics comes from Nashed et al. 2012
        Params
        ======
        action = length 2, applied acceleration in (x, y) directions
        cost = float, cost applied to acting (reflected in reward)
        stay_time = int, number of timesteps agent is required to stay in target box to obtain reward. 1 = immediate reward.
        """
        
        # Add time (in time steps of 10 ms):
        self.time += 1
        dt, kv, tau = 0.01, 1, 0.04

        # Calculate new state using the Newtonian dynamics
        x_pos = self.state[0] + self.state[2]*dt
        y_pos = self.state[1] + self.state[3]*dt
        x_vel = (1-kv*dt) * self.state[2] + dt * self.state[4]
        y_vel = (1-kv*dt) * self.state[3] + dt * self.state[5]
        x_force = (1-dt/tau) * self.state[4] + dt/tau * action[0] + np.random.normal(0., 0.01)# noise level similar to OFC
        y_force = (1-dt/tau) * self.state[5] + dt/tau * action[1] + np.random.normal(0., 0.01)
        
        # Apply forcefield:
        if (self.state[1] + y_vel*dt) < self.ff_lim[0] and (self.state[1] + y_vel*dt) > self.ff_lim[1]:
            x_vel = x_vel + self.ff_force[0]*dt
            y_vel = y_vel + self.ff_force[1]*dt

        # Update position and state
        self.pos = (x_pos, y_pos)
        self.vel = (x_vel, y_vel)
        self.state = np.array([x_pos, y_pos, x_vel, y_vel, x_force, y_force])
        
        # Check if finished 
        if self.goal[0] <= self.pos[0] and self.goal[1] >= self.pos[0]:         # reached goal in x dimension 
            if self.goal[2] <= self.pos[1] and self.goal[3] >= self.pos[1]: # reached goal in y dimension
                self.target_counter += 1
                if self.target_counter >= stay_time:
                    self.reward = 20 - np.linalg.norm(action, 2) * cost # INCREASE FOR SUCCESSFUL TRIALS
                    self.done = True
                
        elif self.bounds[0] <= self.pos[0] or self.bounds[1] >= self.pos[0] or self.bounds[2] >= self.pos[1] \
        or self.bounds[3] <= self.pos[1]: # exited workspace
            self.reward = -10 - np.linalg.norm(action, 2) * cost
            self.done = True 
                
        elif self.time >= self.max_len:     # reached time limit
            self.reward = -10 - np.linalg.norm(action, 2) * cost
            self.reward += self.dist2tar(self,self.pos)*(1-self.discover)
            self.done = True 
            
        else:                             # not finished 
            self.reward = 0 - np.linalg.norm(action, 2) * cost
            self.done = False 
        
        return self 
    
class MultiTarget(TargetReach):
    pass
                           
        
