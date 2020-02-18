import numpy as np

def null_policy(agent, world):  #does not do anything good agent action_callback
    assert agent.good
    agent.action.u = np.zeros(agent.dim_p)
    agent.action.probe = False
    agent.action.tag = False
    return agent.action   #good agent's null policy

def nn_policy_callback(agent, world): #this is used only for getting action, not for belief update
    assert agent.obs is not None  #here the active agent could also be a TD3 actor, depending on the specific type, the callback_policy will handle differently
    raw_action = agent.active_agent.callback_policy(agent.obs, alpha=agent.alpha_action, return_prob=False)
    agent.set_action(raw_action)
    return agent.action  #not necessary to return this action

def neutral_callback(agent, world): #first use a handcraft policy
    raw_action = world.sample_neutral()
    agent.set_action(raw_action)
    return agent.action

def red_ddpg_callback(agent, world):
    assert agent.obs is not None


def agent_action_callback(agent, world):  #when unknown whether neutral or adversary
    if agent.neutral:
        return neutral_callback(agent, world)
    if agent.good or agent.adversary:
        return nn_policy_callback(agent, world)
    raise Exception('neither neutral nor adversary nor good')
