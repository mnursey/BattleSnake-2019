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
import snake_random

if torch.cuda.is_available() and False:
    device = torch.device('cuda')
    print('using cuda')
else:
    device = torch.device('cpu')

class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.h_1 = torch.nn.Linear(n_feature, n_hidden)      # hidden layer
        self.h_2 = torch.nn.Linear(n_hidden, n_hidden)      # hidden layer
        #self.h_3 = torch.nn.Linear(n_hidden, n_hidden)      # hidden layer
        self.output = torch.nn.Linear(n_hidden, n_output)      # hidden layer

    def forward(self, x):
        x = x.float()
        x = F.elu(self.h_1(x))
        x = F.elu(self.h_2(x))
        #x = F.elu(self.h_3(x))
        x = F.softmax(self.output(x), dim=-1)
        return x

class Policy():
    def __init__(self, batch_size, training = False, path=None):
        super(Policy, self).__init__()

        n_input = 2054
        n_hidden = 530
        n_output = 4

        self.gamma = 0.99
        self.learning_rate = 0.0005

        self.net = Net(n_feature = n_input, n_hidden = n_hidden, n_output = n_output).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)

        self.episode = 0
        self.batch_size = batch_size

        self.ep_policy_history = []
        self.ep_reward_episode = []

        self.loss_history = []

        self.training = training

        if training:
            self.net.eval()
        else:
            self.net.train()

        if path != None:
            self.load(path)

        print("Policy: Gamma:{} Learning Rate:{}".format(self.gamma, self.learning_rate))
        print(self.net)

        self.random_chars = random.choice(string.ascii_letters)
        self.random_chars += random.choice(string.ascii_letters)
        self.random_chars += random.choice(string.ascii_letters)
        self.random_chars += random.choice(string.ascii_letters)
        self.random_chars += random.choice(string.ascii_letters)

        print('Random Save ID: ' + self.random_chars)

        return

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
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

        l = loss.item()

        if self.episode % 1000 == 0:
            #print('Trainer: loss history - ' + str(self.loss_history))
            episode_avg = self.loss_history
            episode_avg = sum(episode_avg) / len(episode_avg)
            print('Trainer: episode ' + str(self.episode) + ' ' + str(l) + ' ' + str(episode_avg))
            self.loss_history = []

        if l == 0:
            print("ff - Zero Loss")

        if self.episode % 5000 == 0:
            self.save()

        return l

    def transform_input(self, input):
        window_size = 15

        facing = 0
        my_stats = np.array([input['you']['health'] / 100, len(input['you']['body'])])
        en_stats_health = np.full((window_size, window_size), 0)
        en_stats_length = np.full((window_size, window_size), 0)
        walls = np.full((window_size, window_size), 0)
        friendly = np.full((window_size, window_size), 0)
        my_tail = np.full((window_size, window_size), 0)
        enemy = np.full((window_size, window_size), 0)
        enemy_head = np.full((window_size, window_size), 0)
        en_tails = np.full((window_size, window_size), 0)
        #big_en = np.full((window_size, window_size), 0)
        #sml_en = np.full((window_size, window_size), 0)
        food = np.full((window_size, window_size), 0)
        en_radar = np.full((3, 3), 0)
        tail_radar = np.full((3, 3), 0)
        food_radar = np.full((3, 3), 0)

        centre_x = input['you']['body'][0]['x']
        centre_y = input['you']['body'][0]['y'] 

        # walls
        for x in range(-(window_size // 2), window_size // 2 + 1):
            for y in range(-(window_size // 2), window_size // 2 + 1):
                r_x = centre_x + x
                r_y = centre_y + y

                if 0 > r_x or 0 > r_y or r_x >= input['board']['width'] or r_y  >= input['board']['height']:
                    walls[y + window_size // 2, x + window_size // 2] = 1

        # snakes
        for snake in input['board']['snakes']:
            s_length = len(snake['body'])
            for i, body in enumerate(snake['body']):
                c_x = body['x'] - centre_x
                c_y = body['y'] - centre_y

                if c_x <= window_size // 2 and c_y <= window_size // 2 and c_x >= -window_size // 2 and c_y >= -window_size // 2:
                    if snake['id'] == input['you']['id']:
                        if i == s_length - 1:
                            my_tail[c_y + window_size // 2, c_x + window_size // 2] += 1;
                        else:
                            friendly[c_y + window_size // 2, c_x + window_size // 2] += 1;
                    else:
                        if i == s_length - 1:
                            en_tails[c_y + window_size // 2, c_x + window_size // 2] = 1;

                            if c_x < 0:
                                if c_y < 0:
                                    tail_radar[0, 0] += 1
                                elif c_y > 0:
                                    tail_radar[0, 2] += 1
                                else:
                                    tail_radar[0, 1] += 1
                            elif c_x > 0:
                                if c_y < 0:
                                    tail_radar[2, 0] += 1
                                elif c_y > 0:
                                    tail_radar[2, 2] += 1
                                else:
                                    tail_radar[2, 1] += 1
                            else:
                                if c_y < 0:
                                    tail_radar[1, 0] += 1
                                else:
                                    tail_radar[1, 2] += 1
                        else:
                            enemy[c_y + window_size // 2, c_x + window_size // 2] = 1;

                        if i == 0:
                            enemy_head[c_y + window_size // 2, c_x + window_size // 2] = 1;
                            en_stats_health[c_x + window_size // 2, c_y + window_size // 2] = snake['health'] / 100;
                            en_stats_length[c_x + window_size // 2, c_y + window_size // 2] = len(snake['body'])

                            if c_x < 0:
                                if c_y < 0:
                                    en_radar[0, 0] += 1
                                elif c_y > 0:
                                    en_radar[0, 2] += 1
                                else:
                                    en_radar[0, 1] += 1
                            elif c_x > 0:
                                if c_y < 0:
                                    en_radar[2, 0] += 1
                                elif c_y > 0:
                                    en_radar[2, 2] += 1
                                else:
                                    en_radar[2, 1] += 1
                            else:
                                if c_y < 0:
                                    en_radar[1, 0] += 1
                                else:
                                    en_radar[1, 2] += 1

                            '''if len(snake['body']) < len(input['you']['body']):
                                sml_en[c_x + window_size // 2, c_y + window_size // 2] += 1;
                            else:
                                big_en[c_y + window_size // 2, c_x + window_size // 2] += 1;'''

        # food
        for f in input['board']['food']:
            c_x = f['x'] - centre_x
            c_y = f['y'] - centre_y
            
            if c_x < 0:
                if c_y < 0:
                    food_radar[0, 0] += 1
                elif c_y > 0:
                    food_radar[0, 2] += 1
                else:
                    food_radar[0, 1] += 1
            elif c_x > 0:
                if c_y < 0:
                    food_radar[2, 0] += 1
                elif c_y > 0:
                    food_radar[2, 2] += 1
                else:
                    food_radar[2, 1] += 1
            else:
                if c_y < 0:
                    food_radar[1, 0] += 1
                else:
                    food_radar[1, 2] += 1
                
            if c_x <= window_size // 2 and c_y <= window_size // 2 and c_x >= -window_size // 2 and c_y >= -window_size // 2:
                    food[(c_y + window_size // 2), (c_x + window_size // 2)] = 1;

        # rotate view
        head = input['you']['body'][0]
        neck = input['you']['body'][1]

        # 0 -> north, 1 -> east, 2-> south, 3-> west
        facing = 0
        if neck['y'] < head['y']:
            facing = 2
        if neck['x'] < head['x']:
            facing = 1
        if neck['x'] > head['x']:
            facing = 3

        '''walls = np.rot90(walls, k=facing)
        friendly = np.rot90(friendly, k=facing)
        my_tail = np.rot90(my_tail, k=facing)
        enemy = np.rot90(enemy, k=facing)
        enemy_head = np.rot90(enemy_head, k=facing)
        en_tails = np.rot90(en_tails, k=facing)
        big_en = np.rot90(big_en, k=facing)
        sml_en = np.rot90(sml_en, k=facing)
        food = np.rot90(food, k=facing)
        en_stats_health = np.rot90(en_stats_health, k=facing)
        en_stats_length = np.rot90(en_stats_length, k=facing)'''

        group = np.rot90(np.array([walls, friendly, my_tail, enemy, enemy_head, en_tails, food, en_stats_health, en_stats_length]), k=facing, axes=(1, 2))
        group = np.append(group, np.rot90(np.array([en_radar, tail_radar, food_radar]), k=facing, axes=(1, 2)))
        out = np.append(group.flatten(), my_stats)

        #out = np.append( np.array([walls, friendly, my_tail, enemy, enemy_head, en_tails, big_en, sml_en, food, en_stats_health, en_stats_length]).flatten(), my_stats)
        return out, facing

    def run_ai(self, input):
        model_input, facing = self.transform_input(input)
        model_input = torch.from_numpy(model_input).to(device)
        out = self.net(model_input)
        c = Categorical(out)
        action = c.sample()

        # Add log probability of our chosen action to our history    
        self.ep_policy_history[-1].append(c.log_prob(action))

        action = (action + facing) % 4

        if(action == 0):
            out = 'up'
        elif(action == 1):
            out = 'right'
        elif(action == 2):
            out = 'down'
        elif(action == 3):
            out = 'left'

        return out

    def run_ai_test(self, input, grid=None):
        model_input, facing = self.transform_input(input)
        model_input = torch.from_numpy(model_input).to(device)
        out = self.net(model_input)
        c = Categorical(out)
        action = c.sample()

        action = (action + facing) % 4

        if(action == 0):
            out = 'up'
        elif(action == 1):
            out = 'right'
        elif(action == 2):
            out = 'down'
        elif(action == 3):
            out = 'left'

        if grid is not None:
            f_moves = snake_random.get_free_moves(input, grid)
            f_m_length = len(f_moves)
            if f_m_length > 0 and out not in f_moves:
                out = f_moves[random.randint(0, f_m_length - 1)]

        return out

    def set_reward(self, reward):
        self.ep_reward_episode[-1].append(reward)
        return

    def new_episode(self):
        self.ep_policy_history.append([])
        self.ep_reward_episode.append([])
        return

    def load(self, path):
        saved_state = torch.load(path)
        self.net.load_state_dict(saved_state['state_dict'])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        optimizer.load_state_dict(saved_state['optimizer'])
        return

    def save(self):

        state = {
            'state_dict'    : self.net.state_dict(),
            'optimizer'     : self.optimizer.state_dict(),
            'gamma'         : self.gamma,
            'learning_rate' : self.learning_rate,
            'episode'       : self.episode,
            'random_chars'  : self.random_chars
        }

        path = './models/convnet/' + self.random_chars + str(self.episode) + '.pth'
        torch.save(state, path)

        return path
