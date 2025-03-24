import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class DDPGActor(nn.Module):
    """Actor (Policy) Network for DDPG."""
    
    def __init__(self, state_dim, action_dim, seed, max_action=1.0, hidden_dims=(400, 300)):
        """
        Initialize the Actor network.
        
        Args:
            state_dim (int): Dimension of state space (33 for reacher single arm environment)
            action_dim (int): Dimension of action space (4 for reacher single arm environment)
            seed (int): Random seed
            max_action (float): Maximum action value (default: 1.0)
            hidden_dims (tuple): Dimensions of hidden layers
        """
        super(DDPGActor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.max_action = max_action
        
        # Build the actor network with specified hidden dimensions
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        # Final layer outputs actions
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bounds the action values between -1 and 1 for the continuous control task!
        
        self.model = nn.Sequential(*layers)
        self.reset_parameter()        
        
    def reset_parameter(self):

        # Reset the hidden layers
        for layer in self.model[:-2]:
            if type(layer) == torch.nn.modules.linear.Linear:
                layer.weight.data.uniform_(*hidden_init(layer))
        
        # Reset the output layer
        self.model[-2].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Forward pass to get action values.
        
        Args:
            state (torch.Tensor): Current state
            
        Returns:
            torch.Tensor: Action values scaled to proper range
        """
        return self.max_action * self.model(state)


class DDPGCritic(nn.Module):
    """Critic Network for DDPG."""
    
    def __init__(self, state_dim, action_dim, seed, hidden_dims=(400, 300)):
        """
        Initialize the Critic network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            seed (int): Random seed
            hidden_dims (tuple): Dimensions of hidden layers
        """
        super(DDPGCritic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc0  = nn.Linear(state_dim, hidden_dims[0])
        self.fcs1 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.q_value = nn.Linear(hidden_dims[1], 1)  # Final layer outputs Q value

        self.reset_parameter()

    def reset_parameter(self):

        self.fc0.weight.data.uniform_(*hidden_init(self.fc0))      # 1. hidden layer
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))    # 2. hidden layer + states
        self.q_value.weight.data.uniform_(-3e-4, 3e-4)  # Reset the output layer

    def forward(self, state, action):
        """
        Forward pass to get the estiamted Q-value.
        
        Args:
            state (torch.Tensor): Current state
            action (torch.Tensor): Action
            
        Returns:
            torch.Tensor: Q-Value
        """

        # Process hidden layers
        x = F.relu(self.fc0(state))
        xs = torch.cat((x, action), dim=1)
        x = F.relu(self.fcs1(xs))
        
        return self.q_value(x)
    