# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy

class HLoss(nn.Module): #class used to calculate entropy loss, does not have learning parameters
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = x * torch.log(x)
        b = 1.0 * b.sum(dim=1)
        return b  #return the negative of entropy, so minimize the loss == maximize entropy


class Actor(nn.Module):
    """Actor model

        Parameters:
              args (object): Parameter class
    """

    def __init__(self, state_dim, action_dim, wwid, algo_name):
        super(Actor, self).__init__()
        self.algo_name = algo_name
        if self.algo_name == 'dis':
            self.default_temperature = 1.0

        self.wwid = torch.Tensor([wwid])
        l1 = 400; l2 = 300

        # Construct Hidden Layer 1
        self.f1 = nn.Linear(state_dim, l1)
        self.ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(l1, l2)
        self.ln2 = nn.LayerNorm(l2)

        #Out
        self.w_out = nn.Linear(l2, action_dim)

    def callback_policy(self, state, alpha=None, return_prob=False):  #alpha is meaningless, just to keep the common interface with dqn_agent
        action_prob = self.forward(state).cpu()
        gumbel_sample = self.Gumbel_softmax_sample_distribution(state, use_cuda=False)
        raw_action = int(self.turn_max_into_onehot(gumbel_sample).argmax().numpy().__float__())
        return (raw_action, action_prob) if return_prob else raw_action



    def sample_gumbel(self, shape, use_cuda, eps=1e-20):
        U = torch.rand(shape)
        use_cuda = use_cuda and torch.cuda.is_available()
        return -torch.log(-torch.log(U + eps) + eps).cuda() if use_cuda else -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, use_cuda):
        try:
            y = logits + self.sample_gumbel(logits.size(), use_cuda)
        except RuntimeError:
            y = logits + self.sample_gumbel(logits.size(), use_cuda).cuda()
        return F.softmax(y / temperature, dim=-1)

    def logits(self, input):
        """Method to forward propagate through the actor's graph

            Parameters:
                  input (tensor): states

            Returns:
                  action (tensor): actions
        """
        #Hidden Layer 1
        out = F.elu(self.f1(input))
        out = self.ln1(out)

        #Hidden Layer 2
        out = F.elu(self.f2(out))
        out = self.ln2(out)
        #out = self.w_out(out)
        return out

    def forward(self, input):
        try:
            out = self.w_out(self.logits(input))
        except RuntimeError:
            out = self.w_out(self.logits(input.cuda()))
        #Out
        return F.softmax(out, dim=1) if self.algo_name == "dis" else torch.sigmoid(out)
        #softmax for discrete action

    def Gumbel_softmax_sample_distribution(self, state_batch, temperature = None, use_cuda = False):#a whole vector
        temperature = self.default_temperature if temperature is None else temperature  #if None, use default value, else use the input argument value
        #seems like any pytorch operation on this state_batch tensor cannot be completed
        action_logits = torch.log(self.forward(state_batch))#here we need to have log for the probability!
        return self.gumbel_softmax_sample(action_logits, temperature, use_cuda)

    def turn_max_into_onehot(self, y):
        #shape = y.size()
        if isinstance(y, numpy.ndarray): y = torch.Tensor(y)
        num_classes = y.shape[1]
        labels = y.argmax(dim=1)
        batch_size = labels.size()[0]
        labels_one_hot = torch.FloatTensor(batch_size, num_classes).zero_()
        labels_one_hot.scatter_(1, labels.cpu().unsqueeze(dim=1), 1)
        return labels_one_hot.cuda() if (y.is_cuda and torch.cuda.is_available()) else labels_one_hot

        #return y.argmax() == torch.arange(num_classes).reshape(1, num_classes)
        #return (y.argmax() == torch.arange(num_classes).reshape(1, num_classes)).float()
        #grab from ST-Gumbel-Softmax-Pytorch: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f


    def save_net(self, path):
        torch.save(self.state_dict(), path)

    def load_net(self, path):
        self.load_state_dict(torch.load(path))


class Critic(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        l1 = 400; l2 = 300

        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.q1f1 = nn.Linear(state_dim + action_dim, l1)
        self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q1f2 = nn.Linear(l1, l2)
        self.q1ln2 = nn.LayerNorm(l2)

        #Out
        self.q1out = nn.Linear(l2, 1)


        ######################## Q2 Head ##################
        # Construct Hidden Layer 1 with state
        self.q2f1 = nn.Linear(state_dim + action_dim, l1)
        self.q2ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q2f2 = nn.Linear(l1, l2)
        self.q2ln2 = nn.LayerNorm(l2)

        #Out
        self.q2out = nn.Linear(l2, 1)

        ######################## Value Head ##################  [NOT USED IN CERL]
        # Construct Hidden Layer 1 with
        self.vf1 = nn.Linear(state_dim, l1)
        self.vln1 = nn.LayerNorm(l1)

        # Hidden Layer 2
        self.vf2 = nn.Linear(l1, l2)
        self.vln2 = nn.LayerNorm(l2)

        # Out
        self.vout = nn.Linear(l2, 1)





    def forward(self, obs, action):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """

        #Concatenate observation+action as critic state
        state = torch.cat([obs, action], 1)

        ###### Q1 HEAD ####
        q1 = F.elu(self.q1f1(state))
        q1 = self.q1ln1(q1)
        q1 = F.elu(self.q1f2(q1))
        q1 = self.q1ln2(q1)
        q1 = self.q1out(q1)

        ###### Q2 HEAD ####
        q2 = F.elu(self.q2f1(state))
        q2 = self.q2ln1(q2)
        q2 = F.elu(self.q2f2(q2))
        q2 = self.q2ln2(q2)
        q2 = self.q2out(q2)

        ###### Value HEAD ####
        v = F.elu(self.vf1(obs))
        v = self.vln1(v)
        v = F.elu(self.vf2(v))
        v = self.vln2(v)
        v = self.vout(v)


        return q1, q2, v



# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)

