import sys
import numpy as np
import random
from collections import namedtuple, deque
from models import QNetwork, RNetwork, PolicyNetwork, Classifier
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Agent():
    def __init__(self, state_size, action_size, action_dim, config):
        self.state_size = state_size
        self.action_size = action_size
        self.action_dim = action_dim
        self.seed = 0
        self.device = 'cuda'
        self.batch_size = config["batch_size"]
        self.lr = 0.005
        self.gamma = 0.99
        self.q_shift_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.q_shift_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.Q_local = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.Q_target = QNetwork(state_size, action_size, self.seed).to(self.device)
        self.R_local = RNetwork(state_size,action_size, self.seed).to(self.device)
        self.R_target = RNetwork(state_size, action_size, self.seed).to(self.device)
        self.policy = PolicyNetwork(state_size, action_size,self.seed).to(self.device)
        self.predicter = Classifier(state_size, action_dim, self.seed).to(self.device)
        #self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer_q_shift = optim.Adam(self.q_shift_local.parameters(), lr=self.lr)
        self.optimizer_q = optim.Adam(self.Q_local.parameters(), lr=self.lr)
        self.optimizer_r = optim.Adam(self.R_local.parameters(), lr=self.lr)
        self.optimizer_p = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_pre = optim.Adam(self.predicter.parameters(), lr=self.lr)
        pathname = "lr {} batch_size {} seed {}".format(self.lr, self.batch_size, self.seed)
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
        self.writer = SummaryWriter(tensorboard_name)

    def act(self, state):
        dis, action, log_probs, ent = self.policy.sample_action(torch.Tensor(state).unsqueeze(0))
        return dis, action, log_probs, ent

    def learn(self, memory):
        states, next_states, actions = memory.expert_policy(self.batch_size)
        # actions = actions[0]
        # print("states ",  states)
        self.state_action_frq(states, actions)
        return
        # compute difference between Q_shift and y_sh
        q_sh_value = self.q_shift_local(next_states, actions)
        y_sh = np.empty((self.batch_size,1), dtype=np.float32)
        for idx, s in enumerate(next_states):
            q = []
            for action in all_actions:
                q.append(Q_target(s.unsqueeze(0), action.unsqueeze(0)))
            q_max = max(q)
            np.copyto(y_sh[idx], q_max.detach().numpy())




        y_sh = torch.Tensor(y_sh)
        y_sh *= self.gamma 
        q_shift_loss = F.mse_loss(y_sh, q_shift_values)
        # Minimize the loss
        self.optimizer.zero_grad()
        q_shift_loss.backward()
        self.optimizer.step()

        #minimize MSE between pred Q and y = r'(s,a) + gama * max Q'(s',a)
        q_current = self.Q_local(states, actions)
        r_hat = self.R_target(states, actions)
        # use y_sh as target 
        y_q = r_hat + y_sh 
        
        q_loss = F.mse_loss(q_current, y_q)
        # Minimize the loss
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        
        
        
        
        #  get predicted reward
        r = self.R_local(states, actions)

    def state_action_frq(self, states, action):
        """ Train classifer to compute state action freq
        """

        target = self.predicter(states)
        # target = target
        # create one hot encode y from actions
        #y = torch.LongTensor(self.batch_size, self.action_dim)
        # y = F.one_hot(action.long(), self.action_dim)
        y = action.type(torch.long)
        print("y_hat", target.shape)
        print(action.shape)
        #loss = F.cross_entropy(action.type(torch.long), target.type(torch.long))
        # loss = nn.CrossEntropyLoss()(target, y)
        loss = nn.CrossEntropyLoss()(torch.argmax(target, 1), y)
        #loss = F.cross_entropy(action.type(torch.float), target)
        self.optimizer_pre.zero_grad()
        loss.backward()
        self.optimizer_pre.step()














