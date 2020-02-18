import numpy as np
import random
from collections import namedtuple, deque
from .model import QNetwork, RQNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import itertools


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
DRQN_TRACE_LENGTH = 8   # trace length of the DRQN sample
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QLearningAgent():
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # self.seed = random.seed(seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    #this interface seems not duplicated, check 'agent_default_policy'
    def callback_policy(self, state, alpha=0.5, return_prob=False):  #wrap the specific action_callback_policy, common interface of dqn_agent and TD3 agent actor
        return self.soft_act(state, alpha, return_prob)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)  #drqn use a different replay buffer that neglect the first entry of state

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()      #!sample()should have different behavior
                self._learn(experiences, GAMMA)         #!_learn should have different behavior

    def _learn(self, experiences, gamma):
        raise NotImplementedError

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    #???
    def act(self, state, eps):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #for drqn, state needs to be modified
        state = self._state_mod(torch.from_numpy(state).float().unsqueeze(0))
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    #???
    def soft_act(self, state, alpha=0.5, return_prob=False): #temperature parameter of soft action
        #see dqn_callback for taking action, and naive_policy smart_policy for belief update
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(self._state_mod(state))

        self.qnetwork_local.train()
        action_prob = torch.nn.functional.softmax(action_values *alpha, dim=1).cpu().data.numpy().squeeze()
        action_prob = action_prob / np.sum(action_prob)
        if return_prob:
            return np.random.choice(np.arange(self.action_size), p=action_prob), action_prob
        else:
            return np.random.choice(np.arange(self.action_size), p=action_prob)

    def _state_mod(self):  #modify state by removing the first belief entry,
        if random.random() < 0.0001:
            print("dqn_agent, line 105, only work when we have belief as the first entry of state")
        raise NotImplementedError

    def share_memory(self):
        self.qnetwork_local.share_memory()
        self.qnetwork_target.share_memory()

    def save_net(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    def load_net(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path))
        self.qnetwork_target.load_state_dict(torch.load(path))

    def print_id(self):
        print("my id is:", hex(id(self.qnetwork_local)), hex(id(self.qnetwork_target)))


class DQNAgent(QLearningAgent):
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        super(DQNAgent, self).__init__(state_size, action_size, seed)
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory
        self.memory = DQNReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def _learn(self, experiences, gamma):  #here experiences should be chunks of continuous experience
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self._soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _state_mod(self, state):
        return state

class DRQNAgent(QLearningAgent):

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size - 1
        self.action_size = action_size
        super(DRQNAgent, self).__init__(state_size, action_size, seed)
        # Q-Network
        self.qnetwork_local = RQNetwork(self.state_size, action_size, seed)
        self.qnetwork_target = RQNetwork(self.state_size, action_size, seed)   #not taking the first belief
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory
        self.memory = DRQNReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)


    def _learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences #here experiences is a list
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = [self.qnetwork_target(self._state_mod(next_state), hidden=self.qnetwork_target.init_hidden_layer(), update_hidden=False).detach().max(2)[0].unsqueeze(1) for next_state in next_states]
        # Compute Q targets for current states
        Q_targets = [reward.unsqueeze(-1) + (gamma * Q_tar * (1 - done.unsqueeze(-1))) for reward, Q_tar, done in zip(rewards, Q_targets_next, dones)]
        # Get expected Q values from local model
        Q_expected = [self.qnetwork_local(self._state_mod(state), hidden=self.qnetwork_target.init_hidden_layer(), update_hidden=False).gather(2, action.unsqueeze(-1)) for state, action in zip(states, actions)]
        # Compute loss
        loss = sum([F.mse_loss(Q_expec, Q_tar) for Q_expec, Q_tar in zip(Q_expected, Q_targets)])
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self._soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def _state_mod(self, state):
        return state[:, 1:] if state.shape[1] == self.state_size + 1 else state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        #self.seed = random.seed(seed)

    def add(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DQNReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, seed):
        super(DQNReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        if random.random() <= 0.00005:
            print("dqn_agent line 196 .to(device commented)")
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()  # .to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()  # .to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()  # .to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float()  # .to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()  # .to(device)

        return (states, actions, rewards, next_states, dones)

class DRQNReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, seed):
        super(DRQNReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state[1:], action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        num_trace = self.batch_size // DRQN_TRACE_LENGTH
        trace_start_ind = random.sample(range(len(self.memory)), k=num_trace)
        experiences = [self._truncate(list(itertools.islice(self.memory, start, start + DRQN_TRACE_LENGTH)))
                       for start in trace_start_ind if len(self._truncate(list(itertools.islice(self.memory, start, start + DRQN_TRACE_LENGTH)))) >= 1]
        try:
            states = [torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float() for experience in experiences]
        except ValueError:
            print([len([e.state for e in experience if e is not None]) for experience in experiences])
            import time
            time.sleep(1000)
        actions = [torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).long() for experience in experiences]
        rewards = [torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float() for experience in experiences]
        next_states = [torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float() for experience in experiences] # .to(device)
        dones = [torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float() for experience in experiences]  # .to(device)

        return (states, actions, rewards, next_states, dones)

    def _truncate(self, experience):  # if done, truncate experience
        done_list = [e.done for e in experience]
        for i, done in enumerate(done_list):
            if done:
                experience = experience[:i]
                break
        return experience