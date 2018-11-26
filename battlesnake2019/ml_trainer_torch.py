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

np.set_printoptions(suppress=True)

def json_obj_to_conv_input(json_obj):
    food_plane      = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    my_head_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    my_body_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    my_tail_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    en_head_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    en_body_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    en_tail_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))

    for food in json_obj['board']['food']:
        food_plane[food['y']][food['x']] = 1

    my_id = json_obj['you']['id']

    for snake in json_obj['board']['snakes']:
        if snake['id'] is not my_id:
            en_head_plane[snake['body'][0]['y']][snake['body'][0]['x']] = float(snake['health']) / 100.0
            for body in snake['body']:
                en_body_plane[body['y']][body['x']] = 1
            en_tail_plane[snake['body'][-1]['y']][snake['body'][-1]['x']] = 1

        else:
            my_head_plane[snake['body'][0]['y']][snake['body'][0]['x']] = float(snake['health']) / 100.0
            for body in snake['body']:
                my_body_plane[body['y']][body['x']] = 1
            my_tail_plane[snake['body'][-1]['y']][snake['body'][-1]['x']] = 1

    conv_input = np.array([food_plane, my_head_plane, my_body_plane, my_tail_plane, en_head_plane, en_body_plane, en_tail_plane])
    return conv_input

class ConvNet(nn.Module):
    def __init__(self, board_width, board_height):
        super(ConvNet, self).__init__()

        cuda0 = torch.device('cuda:0')

        self.conv1 = nn.Conv2d(7, 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(128)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv2d(128, 256, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.bn2 = nn.BatchNorm2d(256)
        self.act2 = nn.ELU()

        self.conv3 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ELU()

        self.fc1 = nn.Linear(256 * board_height * board_width, 256) 
        self.act4 = nn.ELU()

        self.output_layer = nn.Linear(256, 4)

        # Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 0.0001

        # Episode policy and reward history 
        self.ep_policy_history = []
        self.ep_reward_episode = []

        # Overall reward and loss history
        self.loss_history = []

        return

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.act4(x)

        x = self.output_layer(x)
        x = torch.softmax(x, dim = -1)

        return x

    def conv3x3(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)

class ConvAI():
    def __init__(self, board_width, board_height, batch_size):
        super(ConvAI, self).__init__()
        
        self.cuda0 = torch.device('cuda:0')
        self.net = ConvNet(board_width, board_height).to(self.cuda0)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.net.learning_rate)

        self.episode = 0

        self.batch_size = batch_size

        self.random_chars = random.choice(string.ascii_letters)
        self.random_chars += random.choice(string.ascii_letters)
        self.random_chars += random.choice(string.ascii_letters)
        self.random_chars += random.choice(string.ascii_letters)
        self.random_chars += random.choice(string.ascii_letters)

        print('Random Save ID: ' + self.random_chars)

        print(self.net)

        return

    def run_ai(self, input, testing):

        if testing:

            self.net.eval()

            net_input =  torch.from_numpy(json_obj_to_conv_input(input)).to(self.cuda0).unsqueeze(0).float()
            out = self.net(net_input)
            values, action = torch.max(out, 1)

            if(action == 0):
                action = 'up'
            if(action == 1):
                action = 'down'
            if(action == 2):
                action = 'left'
            if(action == 3):
                action = 'right'

            self.net.train()

            return action

        net_input =  torch.from_numpy(json_obj_to_conv_input(input)).to(self.cuda0).unsqueeze(0).float()
        out = self.net(net_input)
        c = Categorical(out)
        action = c.sample()

        # Add log probability of our chosen action to our history    
        self.net.ep_policy_history[-1].append(c.log_prob(action))

        if(action == 0):
            action = 'up'
        if(action == 1):
            action = 'down'
        if(action == 2):
            action = 'left'
        if(action == 3):
            action = 'right'

        return action

    def set_reward(self, reward):
        self.net.ep_reward_episode[-1].append(reward)
        return

    def new_episode(self):
        self.net.ep_policy_history.append([]);
        self.net.ep_reward_episode.append([]);
        return

    def update_policy(self):
        R = 0
        rewards = collections.deque()

        # Discount future rewards back to the present using gamma
        for ep_rewards in self.net.ep_reward_episode[::-1]:
            R = 0
            for r in ep_rewards[::-1]:
                R = r + self.net.gamma * R
                rewards.appendleft(R)

        rewards = torch.FloatTensor(rewards).to(self.cuda0)
        #print('rewards')
        #print(rewards)
        if len(rewards) > 1:
            rewards -= rewards.mean()
            if rewards.std() > 0:
                rewards /= rewards.std()
        #print('rewards')
        #print(rewards)

        log_probs = torch.cat([torch.stack(lp) for lp in self.net.ep_policy_history]).transpose(0, 1)
        #print('log probs')
        #print(log_probs)
        policy_loss = torch.mul(-log_probs, rewards)
        #print('policy loss')
        #print(policy_loss)

        self.optimizer.zero_grad()

        # Calculate loss
        loss = policy_loss.sum()
        #print('loss')
        #print(loss)
        # Update network weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        # Save and inialize episode history counters
        self.net.loss_history.append(loss.data.item())
        self.net.ep_reward_episode = []
        self.net.ep_policy_history = []

        self.episode += self.batch_size

        if self.episode % 100 == 0:
            episode_avg = self.net.loss_history
            episode_avg = sum(episode_avg) / len(episode_avg)
            print('Trainer: episode ' + str(self.episode) + ' ' + str(loss.item()) + ' ' + str(episode_avg))
            self.net.loss_history = []

        if self.episode % 5000 == 0:
            self.save()

        return loss.item()

    def load(self):
        saved_state = torch.load('./models/convnet/qDTOS95000.pth')
        self.net.load_state_dict(saved_state['state_dict'])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.net.learning_rate)
        optimizer.load_state_dict(saved_state['optimizer'])

        return

    def save(self):

        state = {
            'state_dict'    : self.net.state_dict(),
            'optimizer'     : self.optimizer.state_dict(),
            'gamma'         : self.net.gamma,
            'episode'       : self.episode,
            'random_chars'  : self.random_chars
        }

        torch.save(state, './models/convnet/' + self.random_chars + str(self.episode) + '.pth')

        return

if __name__ == '__main__':
   run()