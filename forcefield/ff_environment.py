import numpy as np

## HYPERPARAMETERS
ACTION_WEIGHT = 0.5 # effectuates momentum; 
TIME_LIMIT = 50     # max time steps per trial 
COST_PARAM = 0.2 # penalty to action 

class ForceField():
    
    def __init__(self, start_pos = (.5, 1), goal_tl = (.1, 0), goal_br = (.9, -.8), space_padding = 5, max_len=TIME_LIMIT):
        """Initialize forcefield environment.
        Params
        ======
        start_pos: tuple (x, y) coordinates of starting position.
        goal_tl: tuple(x, y) coordinates of top left corner of goal box.
        goal_br: tuple(x, y) coordinates of bottom right corner of goal box.
        space_padding: float space from start_pos in each direction delineating allowed movement.
        """
        
        self.action_size = 2 # (x_velocity, y_velocity)
        self.max_len = max_len
        
        # goal box
        self.start_pos = start_pos 
        self.goal_left = goal_tl[0]
        self.goal_top = goal_tl[1]
        self.goal_right = goal_br[0]
        self.goal_bottom = goal_br[1]
        
        # workspace boundaries
        self.max_top = start_pos[1] + space_padding
        self.max_left = start_pos[0] - space_padding
        self.max_bottom = start_pos[1] - space_padding
        self.max_right = start_pos[0] + space_padding
        
        # forcefields
        self.ff_on = False # no forcefield as default 
        
        
    def add_forcefield(self, top, bottom, force):
        """Add a forcefield. Default is to apply forcefield from a limit in the y dimension.
        Params
        ======
        top: float, y-coord below which (on 2d grid) ff is applied. 
        bottom: float, y-coord above which (on 2d grid) ff is applied. 
        force: (x, y) tuple of forces applied in each direction. 
        """
        
        self.ff_on = True
        self.ff_top = top
        self.ff_bottom = bottom
        self.ff_force = force
        
    def reset(self, pos=(.5, 1)):
        """Reset the environment to the starting position. The start position is (1, .5) on a 2D coordinate system). 
        """
        
        self.pos = pos
        self.state = np.array(list(pos) + [0, 0]) # State is [x_position, y_position, x_velocity, y_velocity].  
        self.next_state = None
        self.reward = 0
        self.done = 0
        self.time = 0
        
        return self 
        
    def step(self, action, act_weight = ACTION_WEIGHT, cost = COST_PARAM):
        """Agent acts in the environment and gets the resulting next state and reward obtained.
        """
        
        # Add time:
        self.time += 1
        
        # Calculate new velocities by weighting with old actions: 
        old_action = self.state[2:]
        x_vel, y_vel = get_carried_action(old_action, action)
        
        # Apply forcefield:
        if self.ff_on:
            if (self.state[1] + y_vel) < self.ff_top and (self.state[1] + y_vel) > self.ff_bottom:
                x_vel = x_vel + self.ff_force 

        # Calculate new positions by adding new velocities to position 
        x_pos = self.state[0] + x_vel
        y_pos = self.state[1] + y_vel
        
        # Update position and state
        self.pos = (x_pos, y_pos)
        self.state = np.array([x_pos, y_pos, x_vel, y_vel])
        
        # Check if finished 
        if self.goal_left <= self.pos[0] and self.goal_right >= self.pos[0]:         # reached goal in x dimension 
            if self.goal_top >= self.pos[1] and self.goal_bottom <= self.pos[1]:     # reached goal in y dimension
                self.reward = 5 - np.linalg.norm(action, 2) * cost
                self.done = True 
                
        elif self.max_top <= self.pos[0] or self.max_bottom >= self.pos[0] or self.max_left >= self.pos[1] \
        or self.max_right <= self.pos[1]: # exited workspace
            self.reward = -5 - np.linalg.norm(action, 2) * cost
            self.done = True 
                
        elif self.time >= self.max_len:     # reached time limit
            self.reward = 0 - np.linalg.norm(action, 2) * cost
            self.done = True 
            
        else:                             # not finished 
            self.reward = 0 - np.linalg.norm(action, 2) * cost
            self.done = False 
        
        return self 
                           
def get_carried_action(old_action, action, act_weight = ACTION_WEIGHT):
        """Get new action based on action from previous timestep. 
        Carried action = (action_weight * old_action) + new_action 
        Params
        ======
        old_action: list length 2 [old_x_velocity, old_y_velocity]
        action: list length 2 [new_x_velocity, new_y_velocity]
        act_weight: weight applied to old velocity 
        """
        
        new_x_velocity = old_action[0] * act_weight + action[0]
        new_y_velocity = old_action[1] * act_weight + action[1]
        
        return new_x_velocity, new_y_velocity                            
                           
        