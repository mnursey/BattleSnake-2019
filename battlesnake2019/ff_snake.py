import os
from sklearn.model_selection import train_test_split
from math import radians
import numpy as np
import pandas as pd
import data_formatter as df
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
from math import radians
import matplotlib.pyplot as plt
import random
import string
import collections

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('using cuda')
else:
    device = torch.device('cpu')    

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.h_1 = torch.nn.Linear(n_feature, n_hidden)      # hidden layer
        self.h_2 = torch.nn.Linear(n_hidden, n_hidden)      # hidden layer
        self.output = torch.nn.Linear(n_hidden, n_output)      # hidden layer

    def forward(self, x):
        x = x.float()
        x = F.relu(self.h_1(x))
        x = F.relu(self.h_2(x))
        x = F.softmax(self.output(x), dim=-1)
        return x

class Policy():
    def __init__(self, board_width, board_height, batch_size):
        super(Policy, self).__init__()

        n_input = 75
        n_hidden = 25
        n_output = 4

        self.gamma = 0.99
        self.learning_rate = 0.001

        self.net = Net(n_feature = n_input, n_hidden = n_hidden, n_output = n_output).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)

        self.episode = 0
        self.batch_size = batch_size

        self.ep_policy_history = []
        self.ep_reward_episode = []

        self.loss_history = []

        print("Policy: Gamma:{} Learning Rate:{}".format(self.gamma, self.learning_rate))
        print(self.net)

        return

    def update_policy(self):
        R = 0
        rewards = collections.deque()

        # Discount future rewards back to the present using gamma
        for ep_rewards in self.ep_reward_episode[::-1]:
            R = 0
            for r in ep_rewards[::-1]:
                R = r + self.gamma * R
                rewards.appendleft(R)

        rewards = torch.FloatTensor(rewards).to(device)

        if len(rewards) > 1:
            rewards -= rewards.mean()
            if rewards.std() > 0:
                rewards /= rewards.std()
        log_probs = torch.cat([torch.stack(lp) for lp in self.ep_policy_history])
        policy_loss = torch.mul(-log_probs, rewards)
        self.optimizer.zero_grad()

        # Calculate loss
        loss = policy_loss.sum()

        # Update network weights
        loss.backward()
        self.optimizer.step()

        # Save and inialize episode history counters
        self.loss_history.append(loss.data.item())
        self.ep_reward_episode = []
        self.ep_policy_history = []

        self.episode += self.batch_size

        if self.episode % 1000 == 0:
            #print('Trainer: loss history - ' + str(self.loss_history))
            episode_avg = self.loss_history
            episode_avg = sum(episode_avg) / len(episode_avg)
            print('Trainer: episode ' + str(self.episode) + ' ' + str(loss.item()) + ' ' + str(episode_avg))
            self.loss_history = []

        '''if self.episode % 5000 == 0:
            self.save()'''

        return loss.item()

    def transform_input(self, input):
        window_size = 5

        facing = 0
        walls = np.full((window_size, window_size), 0)
        friendly = np.full((window_size, window_size), 0)
        enemy = np.full((window_size, window_size), 0)

        centre_x = input['you']['body'][0]['x']
        centre_y = input['you']['body'][0]['y'] 

        for snake in input['board']['snakes']:
            for body in snake['body']:
                c_x = body['x'] - centre_x
                c_y = body['y'] - centre_y

                if c_x <= window_size // 2 and c_y <= window_size // 2 and c_x >= -window_size // 2 and c_y >= -window_size // 2:
                    if snake['id'] == input['you']['id']:
                        friendly[c_x + window_size // 2, c_y + window_size // 2] = 1;
                    else:
                        enemy[c_x + window_size // 2, c_y + window_size // 2] = 1;

        return np.array([walls, friendly, enemy]).flatten(), facing

    def run_ai(self, input):
        model_input, facing = self.transform_input(input)
        model_input = torch.from_numpy(model_input).to(device)
        out = self.net(model_input)
        c = Categorical(out)
        action = c.sample()

        # Add log probability of our chosen action to our history    
        self.ep_policy_history[-1].append(c.log_prob(action))

        if(action == 0):
            action = 'up'
        elif(action == 1):
            action = 'down'
        elif(action == 2):
            action = 'left'
        elif(action == 3):
            action = 'right'

        return action

    def set_reward(self, reward):
        self.ep_reward_episode[-1].append(reward)
        return

    def new_episode(self):
        self.ep_policy_history.append([])
        self.ep_reward_episode.append([])
        return
