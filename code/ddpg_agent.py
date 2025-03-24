import torch
import torch.optim as optim
import numpy as np
import random
import copy
import torch.functional as F

from prioritized_replay_buffer import PrioritizedReplayBufferNStep
from ddpg_architectures import DDPGActor, DDPGCritic 


WEIGHT_DECAY = 0.0001


class DDPGAgent:
    """DDPG agent. Expecially suitedf or continuous control tasks."""
    
    def __init__(self,
                random_seed,
                device_type,
                state_dim,
                action_dim,
                max_action=1.0, 
                hidden_dims=(400, 300),
                lr_actor=1e-4,
                lr_critic=3e-4,
                gamma=0.98,
                tau=1e-3,
                buffer_size=int(1e6),
                batch_size=128,
                n_steps=5,
                learn_every_x_steps=40,
                learning_steps=4) -> None:
        """
        Initialize the DDPG agent.
        
        Args:
            device_type (str): Which backend to use (e.g. CUDA or CPU ...)
            random_seed (int): Common ramdom seed
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            max_action (float): Maximum action value
            hidden_dims (tuple): Dimensions of hidden layers
            lr_actor (float): Learning rate for actor
            lr_critic (float): Learning rate for critic
            gamma (float): Discount factor
            tau (float): Soft update parameter
            buffer_size (int): Replay buffer size
            batch_size (int): Batch size for training
            n_step (int): Number of steps for N-step returns
        """
        self.device_type = device_type
        self.random_seed = random_seed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.n_steps = n_steps
        self.tau = tau
        self.batch_size = batch_size
        self.learn_every_x_steps = learn_every_x_steps
        self.learning_steps: int=learning_steps
        
        # Actor networks (current and target)
        self.actor_local = DDPGActor(state_dim, action_dim, self.random_seed, max_action, hidden_dims).to(self.device_type)
        self.actor_target = DDPGActor(state_dim, action_dim, self.random_seed, max_action, hidden_dims).to(self.device_type)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic networks (current and target)
        self.critic_local = DDPGCritic(state_dim, action_dim, self.random_seed, hidden_dims).to(self.device_type)
        self.critic_target = DDPGCritic(state_dim, action_dim, self.random_seed, hidden_dims).to(self.device_type)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic,  weight_decay=WEIGHT_DECAY)
        
        # Prioritized replay buffer with N-step return estimate
        self.memory = PrioritizedReplayBufferNStep(buffer_size, batch_size, self.device_type, self.random_seed, n_steps, gamma)
                                                       
        # Initialize time step and episode counters for updating/learning
        self.t_step_counter = 0
        self.episode_counter = 0

        self.noise_model = AdaptiveOUNoise(self.action_dim, self.random_seed)


        self.n_steps_discount = self.gamma**self.n_steps

    def reset(self):
        self.episode_counter += 1
        self.noise_model.reset()

    def act(self, state, do_explore=True):
        """
        Choose an action given a state.
        
        Args:
            state: Current state
            add_noise (bool): Whether to add noise for exploration
            
        Returns:
            Action to take, with noise (if demanded) and clipped to allowed value range
        """
        state = torch.from_numpy(state).float().to(self.device_type)
        
        # Prepare "local" actor for evaluation
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        # Set actor back to training mode
        self.actor_local.train()
        
        # Add explorative noise on demand
        if do_explore:
            action += self.noise_model.sample()
         
        # Clip action to be within valid range
        return np.clip(action, -self.max_action, self.max_action)
        
    def step(self, state, action, reward, next_state, done, do_learn):
        """
        Add an experience to the replay buffer (memrory).
        Do a learning at the specified update step and if enough experience is availalble.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps
        self.t_step_counter: int = (self.t_step_counter + 1) % self.learn_every_x_steps
        if (self.t_step_counter == 0) and do_learn:
            for _ in range(self.learning_steps):
                experiences = self.memory.sample()
                if experiences is not None:
                    self.learn(experiences)

    def learn(self, experiences):
        """
        Update actor and critic networks using given batch of experiences with n-step returns.
        Args:
            experiences: Tuple of (states, actions, n_step_rewards, next_states_nth, dones, weights, indices)
        """
        states, actions, n_step_rewards, next_states_nth, dones, weights, indices = experiences
        
        # ---------------------------- Update Critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Note: next_states_nth contains the state after n steps
        actions_next = self.actor_target(next_states_nth)
        Q_targets_next = self.critic_target(next_states_nth, actions_next)
        
        # Compute Q targets for current states (y_i)
        # For n-step returns, use gamma^n as the discount factor for the next state value
        Q_targets = n_step_rewards + (self.n_steps_discount * Q_targets_next * (1 - dones))
        
        # Compute critic loss (using importance sampling weights from PER)
        Q_expected = self.critic_local(states, actions)
        td_errors = Q_targets - Q_expected
        critic_loss = (weights * torch.square(td_errors)).mean()
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self._soft_update(self.critic_local, self.critic_target, self.tau)
        self._soft_update(self.actor_local, self.actor_target, self.tau)
        
        # Update priorities in replay buffer
        with torch.no_grad():
            # Calculate absolute TD errors for priorities (already computed above)
            td_errors_abs = torch.abs(td_errors).detach().cpu().numpy().flatten()
            # Add small constant to avoid zero priority
            priorities = td_errors_abs + 1e-5
            # Update replay buffer priorities
            self.memory.update_priorities(indices, priorities)

    # def learn(self, experiences):
    #     """
    #     Update actor and critic networks using given batch of experiences.
    #     Args:
    #     experiences: Tuple of (states, actions, rewards, next_states, dones, weights, indices)
    #     """
    #     states, actions, rewards, next_states, dones, weights, indices = experiences
        
    #     # ---------------------------- Update Critic ---------------------------- #
    #     # Get predicted next-state actions and Q values from target models
    #     actions_next = self.actor_target(next_states)
    #     Q_targets_next = self.critic_target(next_states, actions_next)
        
    #     # Compute Q targets for current states (y_i)
    #     Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
    #     # Compute critic loss (using importance sampling weights from PER)
    #     Q_expected = self.critic_local(states, actions)
    #     td_errors = Q_targets - Q_expected
    #     critic_loss = (weights * torch.square(td_errors)).mean()
        
    #     # Minimize the loss
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
    #     self.critic_optimizer.step()
        
    #     # ---------------------------- update actor ---------------------------- #
    #     # Compute actor loss
    #     actions_pred = self.actor_local(states)
    #     actor_loss = -self.critic_local(states, actions_pred).mean()
        
    #     # Minimize the loss
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)
    #     self.actor_optimizer.step()
        
    #     # ----------------------- update target networks ----------------------- #
    #     self._soft_update(self.critic_local, self.critic_target, self.tau)
    #     self._soft_update(self.actor_local, self.actor_target, self.tau)
        
    #     # Update priorities in replay buffer
    #     with torch.no_grad():
    #         # Calculate absolute TD errors for priorities (already computed above)
    #         td_errors_abs = torch.abs(td_errors).detach().cpu().numpy().flatten()
            
    #         # Add small constant to avoid zero priority
    #         priorities = td_errors_abs + 1e-5
            
    #         # Update replay buffer priorities
    #         self.memory.update_priorities(indices, priorities)
    
    def reset(self):
        self.episode_counter += 1
        if self.noise_model:
            self.noise_model.reset()

    def _soft_update(self, local_model, target_model, tau):
        """
        Soft update for target networks.
        
        Args:
            local_model: Source model
            target_model: Target model to update
            tau: Interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
    def save(self, filename):
        """
        Save the agent's state.
        
        Args:
            filename: Path to save the agent's state
        """
        torch.save({
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
        
    def load(self, filename):
        """
        Load the agent's state.
        
        Args:
            filename: Path to load the agent's state from
        """
        checkpoint = torch.load(filename)
        self.actor_local.load_state_dict(checkpoint['actor_local'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_local.load_state_dict(checkpoint['critic_local'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


class AdaptiveOUNoise:
    """Adaptive Ornstein-Uhlenbeck process with decaying noise."""
    
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma_start=0.3, sigma_end=0.05, sigma_decay=0.99999):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.sigma_decay = sigma_decay
        self.sigma = sigma_start
        self.size = size
        self.seed = random.seed(seed)
        self.step_counter = 0
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def update_sigma(self, sigma_new):
        self.sigma = sigma_new
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        
        # Decay sigma over time
        self.sigma = max(self.sigma_end, self.sigma * self.sigma_decay)
        
        # Generate noise with current sigma
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        
        self.step_counter += 1
        return self.state
    
    def get_current_sigma(self):
        """Return the current sigma value for monitoring."""
        return self.sigma
