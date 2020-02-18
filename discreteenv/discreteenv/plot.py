from discrete_deception import Scenario
import copy, time
from Statistics import Stat
import pickle
from make_envs import make_naive_adv_env, make_naive_good_env, make_smart_adv_env, make_smart_good_env
import os
#import matplotlib.pyplot as plt
import numpy as np


def tag_index(tag_success):
    return [i for i, item in enumerate(tag_success) if item == True]

def plot_statistics(stat_file_name, env):
    with open(stat_file_name, 'rb') as handle:
        stats = pickle.load(handle)

    fig_path = './pytorch_models/' + env.name + 'stats'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    ep_len = len(stats[0].agent_identity)
    ep = list(range(ep_len))

    fig_index = 0
    tag_success = 0  #successfully tag an adv
    total_adv_ep = 0  #total number of adversary episode
    avg_dis2goal = np.zeros(len(stats[0].agent_identity))
    avg_naive_belief = np.zeros(len(stats[0].agent_identity))
    avg_smart_belief = np.zeros(len(stats[0].agent_identity))
    tag_time = []  #the time steps when blue tags red

    #plot distance to goal
    fig = plt.figure(fig_index)
    for stat in stats:
        if stat.agent_identity[0] == 'adversary':
            total_adv_ep += 1
            if np.any(np.array(stat.tag_success)):
                tag_success += 1
            avg_dis2goal += np.array(stat.dis2goal)
            plt.plot(ep, stat.dis2goal, color='green', linewidth=1.0, alpha=0.3)
            tag_time += tag_index(stat.tag_success)

    avg_dis2goal = avg_dis2goal / total_adv_ep
    print('tag success %i, total adv episodes %i, success rate %.2f' % (
    tag_success, total_adv_ep, tag_success / total_adv_ep))
    plt.plot(ep, avg_dis2goal, color='black', linewidth=2.5, label='Adversary distance to goal')
    plt.xlabel('step')
    plt.ylabel('distance to goal')
    plt.legend()
    fig_name = env.name + '_dis2goal.png'
    fig.savefig(os.path.join(fig_path, fig_name))


    #plot histogram of tag time
    fig_index += 1
    fig = plt.figure(fig_index)
    if len(tag_time) == 0:
        print('never successfully tagged')
    else:
        plt.hist(tag_time, label='time step when blue tags red')
        plt.xlabel('step')
        plt.ylabel('tag count')
        plt.legend()
        fig_name = env.name + '_tag_hist.png'
        fig.savefig(os.path.join(fig_path, fig_name))
        #plt.show()

    #plot naive belief evolution
    if stats[0].naive_belief[0] is not None:
        fig_index += 1
        fig = plt.figure(fig_index)
        for stat in stats:
            if stat.agent_identity[0] == 'adversary':
                plt.plot(ep, stat.naive_belief, color='green', linewidth=1.0, alpha=0.3)
                avg_naive_belief += np.array(stat.naive_belief)
        avg_naive_belief = avg_naive_belief/total_adv_ep
        plt.plot(ep, avg_naive_belief, color='black', linewidth=2.5, label='belief using naive red model')
        plt.xlabel('step')
        plt.ylabel('belief using naive red model')
        plt.legend()
        fig_name = env.name + '_naive_belief.png'
        fig.savefig(os.path.join(fig_path, fig_name))
        #plt.show()

    #plot smart belief evolution
    if stats[0].smart_belief[0] is not None:
        fig_index += 1
        fig = plt.figure(fig_index)
        for stat in stats:
            if stat.agent_identity[0] == 'adversary':
                plt.plot(ep, stat.smart_belief, color='green', linewidth=1.0, alpha=0.3)
                avg_smart_belief += np.array(stat.smart_belief)
        avg_smart_belief = avg_smart_belief/total_adv_ep
        plt.plot(ep, avg_smart_belief, color='black', linewidth=2.5, label='belief using smart red model')
        plt.xlabel('step')
        plt.ylabel('belief using smart red model')
        plt.legend()
        fig_name = env.name + '_smart_belief.png'
        fig.savefig(os.path.join(fig_path, fig_name))

def main():
    seed = 0  #seed here does not matter
    scenario = Scenario()
    env, dqn_agent, policy_agent = make_naive_good_env(scenario, seed, one_agent=True)
    stat_file_name = './pytorch_models/'+env.name +'_stats.pickle' if env.world.dim_p == 1 else './pytorch_models/'+env.name + '_2d' +'_stats.pickle'
    plot_statistics(stat_file_name, env)

if __name__ == "__main__":
    main()




