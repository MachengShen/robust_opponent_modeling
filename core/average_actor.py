import numpy as np
import torch.nn as nn
import torch
from core.models import Actor, HLoss
from torch.optim import Adam

#a wrapper for average behavior of a set of actors, need to define forward()
class AverageActor(Actor):
    def __init__(self, state_dim, action_dim, wwid, algo_name, actor_list, replay_buffer, buffer_gpu, batch_size, iterations):
        super(AverageActor, self).__init__(state_dim, action_dim, wwid, algo_name)
        self.lr = 10e-3
        self.actor_list = actor_list
        self.replay_buffer = replay_buffer
        self.buffer_gpu = buffer_gpu
        self.batch_size = batch_size
        self.iterations = iterations
        self.loss = nn.MSELoss()
        self.HLoss = HLoss()
        self.optim = Adam(self.parameters(), self.lr)

    #train the AverageActor to fit the actor list policies
    def update(self):
        #TODO: use more recent samples?
        for _ in range(self.iterations):
            s, _, a, _, _ = self.replay_buffer.sample(self.batch_size)
            #if not self.buffer_gpu:
            if next(self.parameters()).is_cuda and torch.cuda.is_available():
                s = s.cuda(); a = a.cuda();
            if isinstance(s, list): s = torch.cat(s); a = torch.cat(a);
            a_average = self(s)
            loss = self.loss(a_average, a) + 0.1 * torch.sum(self.HLoss(a_average))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

