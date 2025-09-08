import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """Actor (Policy) Model for MADDPG agent.
    
    The Actor network learns a deterministic policy that maps states to actions.
    It uses a deep neural network with batch normalization for stable training.
    
    Architecture:
        - Input layer: state_size neurons
        - Hidden layer 1: fc1_units neurons with BatchNorm and ReLU
        - Hidden layer 2: fc2_units neurons with ReLU  
        - Output layer: action_size neurons with Tanh activation
        
    The Tanh activation ensures actions are bounded in [-1, 1].
    """

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize Actor network parameters and build model.
        
        Args:
            state_size (int): Dimension of each state observation
            action_size (int): Dimension of each action vector
            seed (int): Random seed for reproducible weight initialization
            fc1_units (int): Number of neurons in first hidden layer
            fc2_units (int): Number of neurons in second hidden layer
            
        Note:
            Uses Xavier uniform initialization for hidden layers and
            small uniform initialization for output layer to ensure
            stable learning at the beginning of training.
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model for MADDPG agent.
    
    The Critic network estimates the Q-value function Q(s,a) for state-action pairs.
    In MADDPG, critics have access to the full global state and all agents' actions
    during training, enabling centralized learning with decentralized execution.
    
    Architecture:
        - Input: concatenated global state (all agents)
        - Hidden layer 1: processes state information with BatchNorm and ReLU
        - Hidden layer 2: processes state+action information with ReLU
        - Output: single Q-value estimate
        
    The network uses batch normalization for stable training and proper
    weight initialization for effective learning.
    """

    def __init__(self, full_state_size, full_action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize Critic network parameters and build model.
        
        Args:
            full_state_size (int): Dimension of global state (all agents combined)
            full_action_size (int): Dimension of global action (all agents combined)
            seed (int): Random seed for reproducible weight initialization
            fcs1_units (int): Number of neurons in first hidden layer
            fc2_units (int): Number of neurons in second hidden layer
            
        Note:
            The first layer processes only state information, while the second
            layer processes the concatenation of first layer output and actions.
            This architecture is typical for DDPG-style critics.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(full_state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + full_action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class MADDPGAgent():
    """Multi-Agent DDPG Agent that interacts and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        
        # Hyperparameters
        self.buffer_size = int(1e6)
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 1e-3
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.weight_decay = 0
        self.update_every = 1
        self.num_updates = 1
        
        # Create agents
        self.agents = []
        for i in range(num_agents):
            agent = DDPGAgent(state_size, action_size, i, 
                            num_agents, random_seed)
            self.agents.append(agent)
        
        # Replay memory (shared among agents)
        self.memory = ReplayBuffer(action_size, self.buffer_size, 
                                  self.batch_size, random_seed)
        
        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed)
        
        self.t_step = 0
        
    def reset(self):
        """Reset noise for all agents."""
        self.noise.reset()
        
    def act(self, states, add_noise=True):
        """Returns actions for all agents given their respective states."""
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, add_noise=False)
            actions.append(action)
        actions = np.array(actions)
        
        if add_noise:
            actions += self.noise.sample()
        
        return np.clip(actions, -1, 1)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory and learn."""
        # Save experience
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(self.num_updates):
                    experiences = self.memory.sample()
                    self.learn(experiences)
    
    def learn(self, experiences):
        """Update value parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to appropriate shapes
        states_full = states.reshape(self.batch_size, -1)
        next_states_full = next_states.reshape(self.batch_size, -1)
        actions_full = actions.reshape(self.batch_size, -1)
        
        # Update each agent
        for agent_idx, agent in enumerate(self.agents):
            # Get agent's state, action, reward, next_state, done
            agent_states = states[:, agent_idx, :]
            agent_actions = actions[:, agent_idx, :]
            agent_rewards = rewards[:, agent_idx].unsqueeze(-1)
            agent_next_states = next_states[:, agent_idx, :]
            agent_dones = dones[:, agent_idx].unsqueeze(-1)
            
            # Get predicted next actions for all agents
            next_actions = []
            for i, other_agent in enumerate(self.agents):
                if i == agent_idx:
                    next_action = agent.actor_target(next_states[:, i, :])
                else:
                    next_action = other_agent.actor_target(next_states[:, i, :])
                next_actions.append(next_action)
            next_actions_full = torch.cat(next_actions, dim=1)
            
            # Update critic
            agent.learn_critic(states_full, actions_full, agent_rewards, 
                              next_states_full, next_actions_full, agent_dones)
            
            # Update actor
            predicted_actions = []
            for i, other_agent in enumerate(self.agents):
                if i == agent_idx:
                    predicted_action = agent.actor_local(states[:, i, :])
                else:
                    predicted_action = other_agent.actor_local(states[:, i, :]).detach()
                predicted_actions.append(predicted_action)
            predicted_actions_full = torch.cat(predicted_actions, dim=1)
            
            agent.learn_actor(states_full, predicted_actions_full)
            
            # Soft update target networks
            agent.soft_update()


class DDPGAgent():
    """Single DDPG Agent."""
    
    def __init__(self, state_size, action_size, agent_id, num_agents, random_seed):
        """Initialize a single Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.agent_id = agent_id
        self.num_agents = num_agents
        
        # Hyperparameters
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.weight_decay = 0
        self.tau = 1e-3
        self.gamma = 0.99
        
        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), 
                                         lr=self.lr_actor)
        
        # Critic Network (takes in all states and actions)
        self.critic_local = Critic(state_size * num_agents, 
                                  action_size * num_agents, 
                                  random_seed).to(device)
        self.critic_target = Critic(state_size * num_agents, 
                                   action_size * num_agents, 
                                   random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                          lr=self.lr_critic, 
                                          weight_decay=self.weight_decay)
        
        # Initialize target networks with local network weights
        self.hard_update()
        
    def hard_update(self):
        """Hard update model parameters."""
        for target_param, local_param in zip(self.actor_target.parameters(), 
                                            self.actor_local.parameters()):
            target_param.data.copy_(local_param.data)
            
        for target_param, local_param in zip(self.critic_target.parameters(), 
                                            self.critic_local.parameters()):
            target_param.data.copy_(local_param.data)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        return action
    
    def learn_critic(self, states, actions, rewards, next_states, next_actions, dones):
        """Update critic network."""
        # Get predicted Q values from target critic
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
    
    def learn_actor(self, states, predicted_actions):
        """Update actor network."""
        # Compute actor loss
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def soft_update(self):
        """Soft update model parameters."""
        for target_param, local_param in zip(self.actor_target.parameters(), 
                                            self.actor_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                   (1.0 - self.tau) * target_param.data)
            
        for target_param, local_param in zip(self.critic_target.parameters(), 
                                            self.critic_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                   (1.0 - self.tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


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
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", 
                                                                "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)