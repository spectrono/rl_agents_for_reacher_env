import numpy as np
import torch
import random

from collections import deque, namedtuple


class PrioritizedReplayBufferNStep:
    """
    Like suggested in D4PG the PRB-N (prioritized replay buffer) supports N-step returns.
    The prioritization helps to more oftenly select samples which are more valuable to the training.
    The N-step returns can help to reduce variance in the expected returns and thus stabilizing the training.
    """

    def __init__(
            self,
            buffer_size,
            batch_size,
            device_type,
            seed,
            n_step,
            gamma=0.99,
            alpha=0.6,
            beta=0.4):
        """
        Initialize the PrioritizedReplayBufferNStep.
        
        Args:
            buffer_size (int): Maximum buffer size
            batch_size (int): Size of each training batch
            device_type: cuda, cpu, etc.
            seed (int): random seed
            n_step (int): Number of steps to consider for N-step returns
                           Setting the value to 1 is like in TD-learning.
            gamma (float): Discount factor
            alpha (float): Priority exponent parameter
            beta (float): Importance sampling exponent parameter
            beta_increment (float): Increment for beta parameter
        """
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device_type = device_type
        self.seed = random.seed(seed)
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.position = 0
        self.size = 0  # Counts how many samples are added to the buffer/memory
        
        # Buffer for storing n_step transitions
        self.n_step_buffer_s_a_ns = deque(maxlen=n_step)
        self.n_step_buffer_rewards = deque(maxlen=n_step)
        self.n_step_buffer_dones = deque(maxlen=n_step)
        self.gamma_cumulative = np.array([self.gamma ** step_idx for step_idx in range(self.n_step)])
        
        # Experience tuple definition
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # Configure beta annealing - exponentially approach 1.0
        self.beta_start = beta
        self.beta_end = 1.0
        self.beta_frames = 100000.0  # Number of frames to anneal beta
        self.frame_count: float = 0.0

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add new transition (SARS-D values) to the rolling n-step buffer (deque)
        self.n_step_buffer_s_a_ns.append((state, action, next_state))
        self.n_step_buffer_rewards.append(reward)
        self.n_step_buffer_dones.append(done)
        
        # Stop here, if we don't have enough transitions for n-step return!
        if len(self.n_step_buffer_rewards) < self.n_step:
            return
                      
        # Get state and action from the first transition
        state  = self.n_step_buffer_s_a_ns[0][0]
        action = self.n_step_buffer_s_a_ns[0][1]
        
        terminal_indices = np.where(np.array(self.n_step_buffer_dones))[0]  # Get indices of terminal states.
        if terminal_indices.size > 0:
            first_terminal_index = terminal_indices[0]  # Get the first terminal index
            n_step_reward = np.sum(self.gamma_cumulative[:first_terminal_index + 1] * self.n_step_buffer_rewards[:first_terminal_index + 1])
            nth_state = self.n_step_buffer_s_a_ns[first_terminal_index][2]
            nth_done  = self.n_step_buffer_dones[first_terminal_index]
        else:
            n_step_reward = np.sum(self.gamma_cumulative * np.array(self.n_step_buffer_rewards))
            nth_state = self.n_step_buffer_s_a_ns[-1][2]
            nth_done  = self.n_step_buffer_dones[-1]

        # Create n-step experience
        e = self.experience(state, action, n_step_reward, nth_state, nth_done)
        
        # Add to replay buffer with max priority for new experiences
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.buffer_size:  # If buffer/memory has space available just add experience sample to memory.
                                          # Add with maximum priority for new experiences (1.0 for the first one!).
            self.memory.append(e)
            self.size += 1
        else:                             # Otherwise replace another experience at self.position
            self.memory[self.position] = e
            
        self.priorities[self.position] = max_priority                # Update priority at position of replaced experience.
        self.position = (self.position + 1) % self.buffer_size  # Move self.positon further and rewind if needed!
        
    def sample(self):
        """
        Sample a batch of experiences from memory with prioritization.
        
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones, weights, indices)
        """

        # If buffer is not filled enough, return None
        if self.size < self.batch_size:
            return None

        # Update beta parameter for importance sampling
        # self.beta = min(1.0, self.beta + self.beta_increment)
        self.beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * (self.frame_count / self.beta_frames))
        self.frame_count += float(self.batch_size)
            
        # Calculate sampling probabilities based on priorities
        priorities_for_all_experiences_in_buffer = self.priorities[:self.size] ** self.alpha
        probs_for_all_experiences_in_buffer = priorities_for_all_experiences_in_buffer / np.sum(priorities_for_all_experiences_in_buffer)
        
        # Sample indices based on priorities
        indices_of_experiences_sampled = np.random.choice(self.size, self.batch_size, replace=False, p=probs_for_all_experiences_in_buffer)
        
        # Calculate importance sampling weights
        weights_for_sampled_experiences = (self.size * probs_for_all_experiences_in_buffer[indices_of_experiences_sampled]) ** (-self.beta)
        weights_for_sampled_experiences /= weights_for_sampled_experiences.max()  # Normalize weights
        
        # Get sampled experiences
        experiences_sampled = [self.memory[idx] for idx in indices_of_experiences_sampled]
        
        # Extract fields from experiences
        states      = torch.from_numpy(np.vstack([e.state      for e in experiences_sampled if e is not None])).float().to(self.device_type)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences_sampled if e is not None])).float().to(self.device_type)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences_sampled if e is not None])).float().to(self.device_type)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences_sampled if e is not None])).float().to(self.device_type)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences_sampled if e is not None]).astype(np.uint8)).float().to(self.device_type)
        
        # Convert weights to tensor
        weights_for_sampled_experiences = torch.from_numpy(weights_for_sampled_experiences).float().to(self.device_type)
        
        return (states, actions, rewards, next_states, dones, weights_for_sampled_experiences, indices_of_experiences_sampled)
    
    def update_priorities(self, indices_of_experiences_sampled, td_errors_abs):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices_of_experiences_sampled (array_like): Indices of sampled experiences
            td_errors_abs (array_like): New priorities for experiences
        """
        
        self.priorities[indices_of_experiences_sampled] = td_errors_abs

    def __len__(self):
        """Return the current size of the memory."""
        return self.size
    