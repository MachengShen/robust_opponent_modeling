from .discrete_action_environment import MultiAgentDisActionEnv
import torch
import numpy as np
from .dqn_agent import DQNAgent, DRQNAgent
from .Action_callback import null_policy, agent_action_callback
from .discrete_deception import Scenario
import os

DEFAULT_SCENARIO = Scenario()
current_path = os.path.dirname(__file__)
MODEL_PATH = current_path + '/pytorch_models/'


def seeding(env: object, seed: object) -> object:
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def _make_default_env(scenario):
    world = scenario.make_world()
    env = MultiAgentDisActionEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        done_callback=scenario.done,
        blue_actual_reward_callback=scenario.blue_actual_reward,
        belief_and_true_type_callback=scenario.belief_and_true_type)
    assert hasattr(env, 'self_play') and env.self_play == False
    return env

#attention: this function must be compatible with CERL
def make_self_play_env(scenario = DEFAULT_SCENARIO, seed = 0, return_policy_agent = False, trainers = None, blue_use_drqn=False):#, actual_red = None): #call this only once, just call env.world.reset()
    env = _make_default_env(scenario)
    env.name = 'self_play'
    env.self_play = True
    #seeding(env, seed)

    _env, dqn_smart_good_agent, dqn_smart_adv,\
    dqn_naive_good_agent, dqn_naive_adv_agent = make_smart_good_env(scenario, seed, return_policy_agent = False)

    for agent in env.world.unknown_agents:
        agent.naive = True
        agent.dqn_agent_naive = dqn_naive_adv_agent if trainers is None else trainers[1]   #this is not unnecessary, since the env will use this to step,
        #although the vairable name is 'dqn', it could actually be TD3 actor
        #if trainers is not None:
        #    agent.TD3_actor = actual_red  #this is used to generate actual samples

    for agent in env.world.good_agents:
        agent.naive = True
        agent.dqn_agent_naive = dqn_naive_good_agent if trainers is None else trainers[0]
        if isinstance(agent.dqn_agent_naive, DRQNAgent):#initialize hidden layer
            agent.dqn_agent_naive.qnetwork_local.init_hidden_layer()
            agent.dqn_agent_naive.qnetwork_target.init_hidden_layer()

    dqn_agents, red_action_dim, red_state_dim = make_DQNagent(env, seed, return_red_dim=True, blue_use_drqn=blue_use_drqn)# if trainers is None else None  #dqn_agent is not binded with env.world
    env.action_dim = red_action_dim
    env.state_dim = red_state_dim

    if return_policy_agent:#this returns all the world agent, since all agents do not have action_callback
        return (env, dqn_agents, env.world.policy_agents) #if (trainers is not None) else (env, dqn_agents[0], env.world.policy_agents) #only return blue dqn
        print("return has been changed, make_envs line 57")
    else:
        return (env, dqn_agents) #if (trainers is not None) else (env, dqn_agents[0]) #only return blue dqn

def make_naive_adv_env(scenario, seed, return_policy_agent = False):
    env = _make_default_env(scenario)

    env.name = 'naive_adv'
    for agent in env.world.adversaries:
        agent.naive = True
    for agent in env.world.good_agents:
        agent.action_callback = null_policy

    #seeding(env, seed)
    dqn_agent = make_DQNagent(env, seed) #the env has specified which agent is policy agent by providing action_callback
    if return_policy_agent:
        return env, dqn_agent, env.world.adversaries[0]
    return env, dqn_agent

def make_naive_good_env(scenario, seed, return_policy_agent = False):
    env = _make_default_env(scenario)
    #assign agent with trained dqn
    env.name = 'naive_good'

    _env, dqn_naive_adv_agent = make_naive_adv_env(scenario, seed)

    for agent in env.world.good_agents:
        agent.naive = True
    for agent in env.world.unknown_agents:
        agent.naive = True
        agent.action_callback = agent_action_callback
        agent.dqn_agent_naive = dqn_naive_adv_agent
        file_name = MODEL_PATH + 'naive_adv' +'.pth' if env.world.dim_p == 1 else MODEL_PATH +'naive_adv'+'_2d' +'.pth'
        #agent.dqn_agent_naive.load_net(file_name)
    if np.random.random() < 0.01:
        print("make_envs line 86, load net commented")

    #seeding(env, seed)
    dqn_naive_good_agent = make_DQNagent(env, seed)

    if return_policy_agent:
        return env, dqn_naive_good_agent, env.world.good_agents[0]

    return env, dqn_naive_good_agent, dqn_naive_adv_agent

def make_smart_adv_env(scenario, seed, return_policy_agent = False):
    env = _make_default_env(scenario)
    env.name = 'smart_adv'

    _env, dqn_naive_good_agent, dqn_naive_adv_agent = make_naive_good_env(scenario, seed)

    for agent in env.world.adversaries:
        agent.naive = False
        agent.dqn_agent_naive = dqn_naive_adv_agent
    for agent in env.world.good_agents:
        agent.naive = True
        agent.action_callback = agent_action_callback
        agent.dqn_agent_naive = dqn_naive_good_agent
        file_name = MODEL_PATH + 'naive_good' + '.pth' if env.world.dim_p == 1 else MODEL_PATH +'naive_good'+'_2d' +'.pth'
        #agent.dqn_agent_naive.load_net(file_name)
    if np.random.random() < 0.01:
        print("make_envs line 111, load net commented")
    #seeding(env, seed)
    dqn_smart_adv_agent = make_DQNagent(env, seed)
    if return_policy_agent:
        return env, dqn_smart_adv_agent, env.world.adversaries[0]
    return env, dqn_smart_adv_agent, dqn_naive_good_agent, dqn_naive_adv_agent

def make_smart_good_env(scenario, seed, return_policy_agent = False):
    env = _make_default_env(scenario)
    env.name = 'smart_good'

    _env, dqn_smart_adv, dqn_naive_good_agent, dqn_naive_adv_agent = make_smart_adv_env(scenario, seed)

    for agent in env.world.unknown_agents:
        agent.naive = False
        agent.action_callback = agent_action_callback
        agent.dqn_agent_naive = dqn_naive_adv_agent
        agent.dqn_agent = dqn_smart_adv
        file_name = MODEL_PATH + 'smart_adv' + '.pth' if env.world.dim_p == 1 else MODEL_PATH + 'smart_adv' + '_2d' + '.pth'
        #agent.dqn_agent.load_net(file_name)
    if np.random.random() < 0.01:
        print("make_envs line 131, load net commented")
    for agent in env.world.good_agents:
        agent.naive = False
        agent.dqn_agent_naive = dqn_naive_good_agent

    #seeding(env, seed)
    dqn_smart_good_agent = make_DQNagent(env, seed)
    if return_policy_agent:
        return env, dqn_smart_good_agent, env.world.good_agents[0]
    return env, dqn_smart_good_agent, dqn_smart_adv, dqn_naive_good_agent, dqn_naive_adv_agent

#take environment and make DQNagent for the policy agent
def make_DQNagent(env, seed, return_red_dim = False, blue_use_drqn=False):
    obs = env.reset()
    dqn_agents = []
    for i, agent in enumerate(env.world.policy_agents):
        state_size = obs[i].shape[0]
        action_size = env.world.policy_agents[i].action_size
        if blue_use_drqn:
            dqn_agents.append(DRQNAgent(state_size, action_size, seed))
        else:
            dqn_agents.append(DQNAgent(state_size, action_size, seed))
    if return_red_dim:
        assert len(dqn_agents) == 2, "should have both blue and red"
        return dqn_agents, env.world.policy_agents[-1].action_size, obs[-1].shape[0]
    else:
        return dqn_agents[0] if len(dqn_agents) == 1 else dqn_agents
