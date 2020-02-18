import numpy as np
from .dqn_agent import DQNAgent
import torch
import random

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        self.p_pos = None
        self.dim_p = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        self.naive_belief = None  #adv agent only have naive_belief
        self.tagged = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.speed = 0.5  #adv or neutral speed
        self.u = None
    @property
    def u_map_2d(self):
        return {0:np.array([0,0]), 1:np.array([-self.speed,0]), 2:np.array([self.speed,0]), 3:np.array([0,-self.speed]), 4:np.array([0,self.speed])}

class GoodAction(Action):
    def __init__(self):
        # physical action
        super(GoodAction, self).__init__()
        self.speed = 1.0
        self.probe = None
        self.tag = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.250
        # color
        self.color = None
        self.state = EntityState()

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

# properties of goal entities
class Goal(Entity):
    def __init__(self):
        super(Goal, self).__init__()

# properties of agent entities
class Agent(Entity):  #adversary
    def __init__(self):
        super(Agent, self).__init__()
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.good = False
        self.adversary = True
        self.neutral = False

        self.pos_range = None
        self.action_size = None  #size of available actions, used initialize dqn agent
        self.obs = None  #this is the agent observation, used to take action for neural policy agent
        self.obs_naive = None  #observation if use naive agent mode
        self.dqn_agent = None  #bind the agent with its dqn net
        self.dqn_agent_naive = None
        self.raw_action = None
        self.action_prob = None
        self.naive = None  #
        self.alpha_action = 2.0  #soft_max temperature of taking actual action
        self.alpha_belief_update = 0.5  #temperature for belief update, need to be soft enough
        self.pomdp_adv = False

    def turn2pomdp_adv(self):
        assert self.adversary, "must be an adversary"
        self.pomdp_adv = True
        self.pomdp_adv_policy = np.array([0.1, 0.1, 0.5, 0.05, 0.25])

    @property
    def action_decoder(self):
        if self.dim_p == 1:
            return ['stay', 'forward', 'fast_forward'] if self.adversary else ['left', 'stay', 'right', 'probe', 'tag']
        if self.dim_p == 2:
            return ['stay', 'left', 'right', 'down', 'up'] if self.adversary else ['stay', 'left', 'right', 'down', 'up', 'probe', 'tag']

        raise Exception('no valid action_decoder, check object attribute')

    @property
    def belief(self):
        if self.good:
            return self.state.naive_belief if self.naive == True else self.state.smart_belief
        if self.adversary:
            return None if self.naive == True else self.state.naive_belief

    def set_belief(self, value):
        if self.good:
            if self.naive == True: self.state.naive_belief = value;
            else: self.state.smart_belief = value;
        if self.adversary:
            if self.naive == True: raise Exception("naive adv does not have belief")
            else: self.state.naive_belief = value

    @property
    def active_agent(self):
        assert self.naive is not None
        if self.naive:
            return self.dqn_agent_naive
        else:
            return self.dqn_agent

    def default_policy(self, obs):
        if self.pomdp_adv:
            return np.random.choice(list(range(self.pomdp_adv_policy.shape[0])), size=1, p=self.pomdp_adv_policy)[0]
        else:
            assert not self.neutral, "neutral default policy defined in world"
            assert self.adversary, "this virtual method only work for adversary"
            gumbel_sample = self.TD3_actor.Gumbel_softmax_sample_distribution(torch.from_numpy(obs).float().unsqueeze(0))
            try:return int(self.TD3_actor.turn_max_into_onehot(gumbel_sample).argmax().numpy().__float__());
            except TypeError: return int(self.TD3_actor.turn_max_into_onehot(gumbel_sample).argmax().cpu().numpy().__float__())

    def naive_policy(self, obs, return_prob=False):  #this is used only for belief update
        if self.pomdp_adv:
            return self.default_policy(obs) if not return_prob else (self.default_policy(obs), self.pomdp_adv_policy)
        if isinstance(self.dqn_agent_naive, DQNAgent):
            return self.dqn_agent_naive.soft_act(obs, alpha=self.alpha_belief_update, return_prob=return_prob)
        else:
            #prob = self.dqn_agent_naive.Gumbel_softmax_sample_distribution(torch.from_numpy(obs).float().unsqueeze(0)).detach().numpy().flatten().cuda()
            prob = self.dqn_agent_naive(torch.from_numpy(obs).float().unsqueeze(0)).detach().numpy().flatten()
            sample = np.random.choice(np.arange(prob.shape[0]), p=prob)
            return (sample, prob) if return_prob else sample  #in this case, the dqn_agent is actually Actor in CERL

    def smart_policy(self, obs, return_prob=False): #this is used only for belief update
        return self.dqn_agent.soft_act(obs, alpha=self.alpha_belief_update, return_prob=return_prob)

    def set_action(self, raw_action):  #raw action is the action output of neural network (discrete for dqn)
        self.raw_action = raw_action
        assert raw_action < self.action_size
        self.action.u = np.zeros(self.dim_p)
        if self.dim_p == 1:
            self.action.u = np.array([float(raw_action)]) #stay, forward, or fastforward
        else:
            assert self.dim_p == 2
            self.action.u = self.action.u_map_2d[raw_action]

    def sample_pos(self):
        if self.good:
            x = np.random.uniform(low=self.pos_range[0][0], high=self.pos_range[0][1])
        else:
            assert self.adversary or self.neutral
            x = np.random.uniform(low=self.pos_range[0][0], high=self.pos_range[0][1]) if self.dim_p == 2 else np.random.uniform(low=self.pos_range[0][0], high=self.pos_range[0][1]/3.0)
        self.state.p_pos = np.array([x])
        if self.dim_p == 2:
            if self.good:
                y = np.random.uniform(low=self.pos_range[1][0], high=self.pos_range[1][1])
            else:
                assert self.adversary or self.neutral
                y = np.random.uniform(low=self.pos_range[1][0], high=self.pos_range[1][1]/3.0)
            self.state.p_pos = np.array([x, y])

class GoodAgent(Agent):
    def __init__(self):
        super(GoodAgent, self).__init__()
        self.good = True
        self.adversary = False
        self.neutral = False
        self.action = GoodAction()
        self.state.naive_belief = None
        self.state.smart_belief = None
        self.state.tag_count = np.array([0])
        self.state.probe_count = np.array([0])

    def default_policy(self, obs):
        return self.dqn_agent_naive.soft_act(obs)

    def set_action(self, raw_action):  #raw action is the action output of neural network (discrete for dqn)
        assert raw_action < self.action_size
        self.raw_action = raw_action
        self.action.probe = False
        self.action.tag = False
        self.action.u = np.zeros(self.dim_p)
        if self.dim_p == 1:
            if raw_action <= 2:
                self.action.u = np.array([float(raw_action)]) - 1.0 #move left, stay or right
            if raw_action == 3:
                self.action.probe = True
            if raw_action == 4:
                self.action.tag = True
        else:
            assert self.dim_p == 2
            if raw_action <= 4:
                self.action.u = self.action.u_map_2d[raw_action]
            if raw_action == 5:
                self.action.probe = True
            if raw_action == 6:
                self.action.tag = True




