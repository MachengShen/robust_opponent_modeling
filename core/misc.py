#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import torch
import copy
import logging
import os.path

#true positive, false positive, true negative, false negative
def log_TPR_FPR_TNR_FNR(belief_type_list, prefix=None):
    tpr_10, fpr_10, tnr_10, fnr_10, tpr_20, fpr_20, tnr_20, fnr_20 = [], [], [], [], [], [], [], []
    #tpr + fnr == 1, fpr + tnr == 1
    for i, b_ts in enumerate(belief_type_list):
        if b_ts[5].true_type == 1:  #bug, should not use b_ts[0].true_type
            tpr_10.append(np.mean([b_t.belief for b_t in b_ts][:10]))
            tpr_20.append(np.mean([b_t.belief for b_t in b_ts][:20]))
            fnr_10.append(np.mean([1 - b_t.belief for b_t in b_ts][:10]))
            fnr_20.append(np.mean([1 - b_t.belief for b_t in b_ts][:20]))
        else:
            fpr_10.append(np.mean([b_t.belief for b_t in b_ts][:10]))
            fpr_20.append(np.mean([b_t.belief for b_t in b_ts][:20]))
            tnr_10.append(np.mean([1 - b_t.belief for b_t in b_ts][:10]))
            tnr_20.append(np.mean([1 - b_t.belief for b_t in b_ts][:20]))

    TPR_10, TPR_20, FPR_10, FPR_20 = np.mean(tpr_10), np.mean(tpr_20), np.mean(fpr_10), np.mean(fpr_20)
    TNR_10, TNR_20, FNR_10, FNR_20 = np.mean(tnr_10), np.mean(tnr_20), np.mean(fnr_10), np.mean(fnr_20)
    msg = "TPR_10: " + str(TPR_10)[:5] + ", TPR_20: " + str(TPR_20)[:5] + ", TNR_10: " + str(TNR_10)[:5] + ", TNR_20: " + str(TNR_20)[:5]
    if prefix:
        msg = prefix + msg
    logging.info(msg)
    return TPR_10, TPR_20, FPR_10, FPR_20, TNR_10, TNR_20, FNR_10, FNR_20




def initialize_logger(output_dir, file_name, google_drive_dir=None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    """
    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, "error_" + str(id) + ".log"), "w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    """
    # create info file handler and set level to info
    handler = logging.FileHandler(os.path.join(output_dir, file_name), "w")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if google_drive_dir is not None:
        handler = logging.FileHandler(os.path.join(google_drive_dir, file_name), "w")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

def resolution(range, n):
    return (range[1] - range[0]) / n

def print_Q(critic, obs, actor):
    Q = []
    action = actor(obs)  #only used to get a valid action tensor
    for i in range(action.shape[1]):
        action = action * 0
        action[0][i] = 1.0
        q, _, _ = critic(obs, action)
        Q.append(q)
    print("Q for each action", Q)

#take in learner (correspond to red agent), env, and number of grid
#plot red agent Q function heatmap
def visualize_critic(learner, env, N_GRID):
    agent = env.world.agents[1]  #plot adversary agent
    actor = learner.algo.actor
    critic = learner.algo.critic
    x_range = env.world.x_range
    y_range = env.world.y_range
    x_resolution = resolution(x_range, N_GRID)
    y_resolution = resolution(y_range, N_GRID)
    agent_pos = [[[x_range[0] + x_resolution*i, y_range[0] + y_resolution*j] for j in range(N_GRID)] for i in range(N_GRID)]
    Q = np.zeros([N_GRID, N_GRID])
    plt.figure(figsize=(15, 15))
    for i in range(N_GRID):
        for j in range(N_GRID):
            agent.state.p_pos = np.array(agent_pos[i][j])
            obs = torch.FloatTensor(env.observation_callback(agent, env.world)).unsqueeze(0)
            if torch.cuda.is_available(): obs.cuda();
            #action = torch.FloatTensor(np.zeros(len(agent.action_decoder))).unsqueeze(0).cuda()#to(device)  #shall we use policy instead?
            action = actor(obs)
            q1, q2, val = critic(obs, action)  #val seems not used at all
            #Q[j,i] = (q1.cpu().data.numpy() + q2.cpu().data.numpy()) / 2.0  #critic return 3 quantity: q1,q2,V  (TD3)
            Q[j, i] = q1.cpu().data.numpy()  # critic return 3 quantity: q1,q2,V  (TD3)
            P = action.cpu().data.numpy()/2.0
            if i % 4 == 0 and j % 4 == 0:
                #print("P is ", P)
                #print_Q(critic, obs, actor)
                for k in range(len(agent.action_decoder)):
                    if agent.action_decoder[k] == 'left':
                        x = -P[0][k]; y = 0
                        plt.arrow(agent_pos[i][j][0], agent_pos[i][j][1], x, y, head_width=0.075, head_length=0.075, fc='k', ec='k')
                    if agent.action_decoder[k] == 'right':
                        x = P[0][k]; y = 0
                        plt.arrow(agent_pos[i][j][0], agent_pos[i][j][1], x, y, head_width=0.075, head_length=0.075, fc='k', ec='k')
                    if agent.action_decoder[k] == 'up':
                        x = 0; y = P[0][k]
                        plt.arrow(agent_pos[i][j][0], agent_pos[i][j][1], x, y, head_width=0.075, head_length=0.075, fc='k', ec='k')
                    if agent.action_decoder[k] == 'down':
                        x = 0; y = -P[0][k]
                        plt.arrow(agent_pos[i][j][0], agent_pos[i][j][1], x, y, head_width=0.075, head_length=0.075, fc='k', ec='k')
    cp = plt.contour([x_range[0] + x_resolution*i for i in range(N_GRID)], [y_range[0] + y_resolution*i for i in range(N_GRID)], Q, 20)
    plt.clabel(cp, inline=True, fontsize=10)
    plt.axis('equal')
    plt.show()

