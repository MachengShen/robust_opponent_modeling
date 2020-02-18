import torch
import argparse
import os
import numpy as np
#from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from discreteenv.discrete_deception import Scenario
from discreteenv.discrete_action_environment import MultiAgentDisActionEnv
from discreteenv.dqn_agent import DQNAgent
import copy, time
from discreteenv.Statistics import Stat
import pickle
from discreteenv.make_envs import make_naive_adv_env, make_naive_good_env, make_smart_adv_env, make_smart_good_env, make_self_play_env
from termcolor import colored

def main():
    seed = 10
    Max_step = 25
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.999 #was 0.995
    with_neutral = True
    restore = False  #when training, had better restore == False
    save = True
    Training = True
    Max_episode = 5000 if Training else 30
    TRAIN_GOOD = False
    TRAIN_ADV = True
    current_path = os.path.dirname(__file__)

    # Create env
    scenario = Scenario()
    env, dqn_agents, policy_agents = make_self_play_env(scenario=scenario, seed=seed, return_policy_agent=True) #make self play_env has a default scenario
    if not isinstance(dqn_agents, list):
        dqn_agents = [dqn_agents]
    if len(policy_agents) == 1:
        policy_agents = [policy_agents]

    if not Training:
        env.world.random_pose = False  #deactivate random_pose during testing
    if restore:
        for i, dqn_agent in enumerate(dqn_agents):
            file_name = current_path + '/discreteenv/pytorch_models/' + env.name + '_agent_' + str(i) + '.pth' if env.world.dim_p == 1 else current_path + '/discreteenv/pytorch_models/' + env.name + '_2d_agent_' + str(i) + '.pth'
            dqn_agent.load_net(file_name)

    eps = eps_start
    stats = []
    for i in range(Max_episode):
        stat = Stat()
        ep_rew = 0 if env.name == 'naive_adv' else np.zeros(len(policy_agents))
        # use neutral agent if in self play mode or
        if with_neutral and (env.self_play or (not env.self_play and policy_agents[0].good)):  #randomly select neutral or adversary
            env.world.randomize_neu_adv(neutral_prob = 0.5)
        obs_n = env.reset()
        for j in range(Max_step):
            action_n = []
            action_n_stat = []
            if Training:
                #get actions from agent policies
                for k, dqn_agent in enumerate(dqn_agents):
                    if policy_agents[k].neutral:
                        action_n.append(env.world.sample_neutral())
                    else:
                        action_n.append(dqn_agent.act(obs_n[k], eps=eps))
                    #sad add_tagged_state(action_n_stat, action_n, env) move after step
                if i % 100 == 0:
                    env.render()  # render is slow
                    time.sleep(0.1)
                    ##add some comments
            else:
                #action = dqn_agent.act(obs_n[0], eps=0)
                for k, dqn_agent in enumerate(dqn_agents):
                    if policy_agents[k].neutral:
                        action_n.append(env.world.sample_neutral())
                    else:
                        action_n.append(dqn_agent.soft_act(obs_n[k], alpha=10.0))
                    #add_tagged_state(action_n_stat, action_n, env)
                    env.render()  #render is slow
                    time.sleep(0.1)

            new_obs_n, reward_n, done_n, _ = env.step(copy.deepcopy(action_n))
            add_tagged_state(action_n_stat, action_n, env)

            if Training:
                for k, dqn_agent in enumerate(dqn_agents):
                    #if not policy_agents[k].neutral: #not train neutral
                    if (TRAIN_GOOD and policy_agents[k].good) or (TRAIN_ADV and policy_agents[k].adversary):
                        dqn_agent.step(obs_n[k], action_n[k], reward_n[k], new_obs_n[k], done_n[k])

            obs_n = new_obs_n

            print_info(env, action_n_stat)
            if not Training:
                env.world.collect_statistics(stat)

            ep_rew = add_reward(ep_rew, env, reward_n)
            print_step_rew(reward_n)
        eps = max(eps_end, eps_decay * eps)
        print_rew(env, i, ep_rew, policy_agents)  #check if this is correct
        if not Training:
            stats.append(stat)

    if not Training:
        stat_file_name = current_path + '/discreteenv/pytorch_models/' +env.name +'_stats.pickle' if env.world.dim_p == 1 else current_path + '/discreteenv/pytorch_models/' + env.name + '_2d' + '_stats.pickle'
        with open(stat_file_name, 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if save:
        for i, dqn_agent in enumerate(dqn_agents):
            file_name = current_path + '/discreteenv/pytorch_models/'+env.name + '_agent_' + str(i) +'.pth' if env.world.dim_p == 1 else current_path + '/discreteenv/pytorch_models/'+env.name + '_2d_agent_' + str(i) +'.pth'
            dqn_agent.save_net(file_name)


def add_tagged_state(action_n_stat, action_n, env):  #add tagged state to action_n_stat
    for action in action_n:
        action_n_stat.append((action, env.world.get_tagged_state()))

def print_step_rew(reward_n):
    print(colored('Reward good:','blue'), '%.3f' %(reward_n[0]))
    if reward_n[1] is not None:
        print(colored('Reward adv:','red'), '%.3f' %(reward_n[1]))


def print_info(env, action_n_sta):
    action_n = env.world.action_n()  #decoded action
    color = ['white', 'white'] #good agent font color and unknown agent color
    if action_n[0] == 'tag':
        if action_n_sta[0][1]:
            color[0] = 'green'
            if env.world.unknown_agents[0].neutral:
                color[1] = 'red'
            else:
                color[1] = 'green'
        else:
            color[0] = 'red'
    if action_n[0] == 'probe':
        color[0] = 'blue'
        color[1] = 'red' if env.adversary_opponent else 'blue'

    if env.world.good_agents[0].naive == False:
        print(colored(action_n[0], color[0]), colored(action_n[1], color[1]), "belief, naive %.3f, smart %.3f" % (
        env.world.good_agents[0].state.naive_belief, env.world.good_agents[0].state.smart_belief))
    elif env.world.good_agents[0].naive == True:
        print(colored(action_n[0], color[0]), colored(action_n[1], color[1]), "belief, naive %.3f" % (env.world.good_agents[0].state.naive_belief))

def print_rew(env, ep_count, ep_rew, policy_agents):
    if None in ep_rew:
        print("episode %i, eps_rew %.3f " % (ep_count, ep_rew[0]))
    elif not env.name == 'naive_adv':
        print("episode %i, eps_rew %.3f, %.3f " % (ep_count, ep_rew[0], ep_rew[1]))
    else:
        print("episode %i, eps_rew %.3f" % (ep_count, ep_rew))

    for i, policy_agent in enumerate(policy_agents):
        if policy_agent.naive == True:
            print("final belief, naive %.3f" % env.world.good_agents[0].state.naive_belief)
        if policy_agent.naive == False and policy_agent.good:
            print("final belief, naive %.3f, smart %.3f" % (
            env.world.good_agents[0].state.naive_belief, env.world.good_agents[0].state.smart_belief))

def add_reward(ep_rew, env, reward_n):
    if env.name == 'naive_adv':
        ans = ep_rew + reward_n[0]
    else:
        ans = copy.deepcopy(reward_n)
        for i, rew in enumerate(reward_n):
            ans[i] = (ep_rew[i] + ans[i]) if reward_n[i] is not None else None
    return ans

if __name__ == "__main__":
    main()