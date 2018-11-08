import os
from sklearn.model_selection import train_test_split
from math import radians
import numpy as np
import pandas as pd
import data_formatter as df
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn as nn
from math import radians
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def run():
    #run_test_one('./formatted_data/1vs1_10by10_1f_2018snake.csv')
    load_and_test('./formatted_data/1vs1_10by10_1f_2018snake.csv')
    return

def load_and_test(data_loc): 
    print('Trainer: setting up for testing')

    cuda0 = torch.device('cuda:0')

    raw_data = pd.read_csv(data_loc, header = 0)
    input_raw = raw_data.iloc[:,4:]
    labels_raw = raw_data.iloc[:,:4]

    # Print Input and Labels
    print('Trainer: inputs')
    print(input_raw.head(3))
    print('Trainer: labels')
    print(labels_raw.head(3))

    X_train, X_test, y_train, y_test = train_test_split(input_raw, labels_raw, test_size= 0.1)
        
    X_train = np.float32(X_train)
    y_train = np.float32(y_train)

    # TEMP
    # convert_to_conv(X_train[0])

    X_train = torch.from_numpy(X_train).cuda()
    y_train = torch.from_numpy(y_train).cuda()

    print(X_test)
    print(y_test)

    X_test = np.float32(X_test)
    y_test = np.float32(y_test)

    X_test = torch.from_numpy(X_test).cuda()
    y_test = torch.from_numpy(y_test).cuda()

    # hyperparameters
    n_input = 44
    n_hidden = 150
    n_output = 4
    learning_rate = 0.004
    momentum = 0.9
    epochs = 5000

    net = Net(n_feature = n_input, n_hidden = n_hidden, n_output = n_output)
    net.load_state_dict(torch.load('./models/test1/sigmoid3longtrain.pt'))
    net = net.cuda()
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    plt.axis([0, 1, 0, 1])

    # plt.ion() # interactive mode for graphing

    net.eval()
    out = net(X_test)
    loss = loss_func(out, torch.max(y_test.float(), 1)[1])
    preds = torch.max(out, 1)[1]
    labs = torch.max(y_test.float(), 1)[1]

    print('-' * 5)
    print('Trainer: t_out - ' + str(preds))
    print('Trainer: t_lab - ' + str(labs))
    print('Trainer: t_cor - ' + str( (labs.eq(preds)).sum() ) )
    t_acc = float((labs.eq(preds)).sum()) / y_test.size()[0]
    print('Trainer: t_acc - ' + str(t_acc))
    print('Trainer: t_los - ' + str(loss))
    plt.plot([0],[t_acc], 'go')
    print('-' * 15)

    print('Trainer: testing finished')
    plt.show()

    return

def run_test_one(data_loc):

    print('Trainer: setting up for training')

    cuda0 = torch.device('cuda:0')

    raw_data = pd.read_csv(data_loc, header = 0)
    input_raw = raw_data.iloc[:,4:]
    labels_raw = raw_data.iloc[:,:4]

    np.set_printoptions(suppress=True)

    # Print Input and Labels
    print('Trainer: inputs')
    print(input_raw.head(3))
    print('Trainer: labels')
    print(labels_raw.head(3))

    X_train, X_test, y_train, y_test = train_test_split(input_raw, labels_raw, test_size=0.2)
        
    X_train = np.float32(X_train)
    y_train = np.float32(y_train)

    X_train = torch.from_numpy(X_train).cuda()
    y_train = torch.from_numpy(y_train).cuda()

    print(X_train)
    print(y_train)

    X_test = np.float32(X_test)
    y_test = np.float32(y_test)

    X_test = torch.from_numpy(X_test).cuda()
    y_test = torch.from_numpy(y_test).cuda()

    # hyperparameters
    n_input = 44
    n_hidden = 150
    n_output = 4
    learning_rate = 0.004
    momentum = 0.9
    epochs = 8000

    net = Net(n_feature = n_input, n_hidden = n_hidden, n_output = n_output)
    net = net.cuda()
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    plt.axis([0, 1, 0, 1])

    # plt.ion() # interactive mode for graphing

    for step in range(epochs):
        out = net(X_train)
        loss = loss_func(out, torch.max(y_train.float(), 1)[1])
        preds = torch.max(out, 1)[1]
        labs = torch.max(y_train.float(), 1)[1]

        stp_p = float((step + 1)) / float(epochs)

        if step % 100 == 0 or step + 1 == epochs:
            print('-' * 15)
            print('Trainer: stp - ' + str(step) + ' ' + str(stp_p))
            print('Trainer: out - ' + str(preds))
            print('Trainer: lab - ' + str(labs))
            print('Trainer: cor - ' + str( (labs.eq(preds)).sum() ) )
            acc = float((labs.eq(preds)).sum()) / y_train.size()[0]
            print('Trainer: acc - ' + str(acc))
            print('Trainer: los - ' + str(loss))
            plt.plot([stp_p],[acc], 'ro')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step + 1 == epochs:
            net.eval()
            out = net(X_test)
            loss = loss_func(out, torch.max(y_test.float(), 1)[1])
            preds = torch.max(out, 1)[1]
            labs = torch.max(y_test.float(), 1)[1]

            print('-' * 5)
            print('Trainer: t_out - ' + str(preds))
            print('Trainer: t_lab - ' + str(labs))
            print('Trainer: t_cor - ' + str( (labs.eq(preds)).sum() ) )
            t_acc = float((labs.eq(preds)).sum()) / y_test.size()[0]
            print('Trainer: t_acc - ' + str(t_acc))
            print('Trainer: t_los - ' + str(loss))
            plt.plot([stp_p],[t_acc], 'go')
            print('-' * 15)
            net.train()


    print('Trainer: training finished')

    plt.show()

    torch.save(net.state_dict(), './models/test1/model4.pt')

    return

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.in_ = torch.nn.Linear(n_feature, n_hidden)     # input layer
        self.h_1 = torch.nn.Linear(n_hidden, n_hidden)      # hidden layer
        self.h_2 = torch.nn.Linear(n_hidden, n_hidden)      # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)      # output layer


    def forward(self, x):
        x = self.in_(x.float())  # activation function for hidden layer
        x = torch.sigmoid(self.h_1(x))
        x = torch.sigmoid(self.h_2(x))
        x = torch.sigmoid(self.out(x))
        return x

class ModelRunner():
    cuda0 = torch.device('cuda:0')

    def __init__(self, model_loc):

        print('Trainer: loading model for testing')

        n_input = 44
        n_hidden = 150
        n_output = 4

        net = Net(n_feature = n_input, n_hidden = n_hidden, n_output = n_output)
        net.load_state_dict(torch.load('./models/test1/sigmoid3longtrain.pt'))
        self.net = net.cuda()

        print(net)
        self.net.eval()

        print('Trainer: model loaded')

        return

    def run(self, input):
        model_input = df.simple_obj_to_format(input)
        model_input = torch.from_numpy(model_input).cuda()
        out = self.net(model_input)
        out = torch.max(out, dim = 0, keepdim = True)[1]
        # print('Trainer: model output - ' + str(out))
        return out


def convert_to_conv(np_data): 

    shape = np.shape(np_data)
    num_rows = shape[0]
                
    # food
    # my head
    # my body
    # enemy head
    # enemy body

    # five channels

    food_plane      = np.zeros((10, 10))
    my_head_plane   = np.zeros((10, 10))
    my_body_plane   = np.zeros((10, 10))
    en_head_plane   = np.zeros((10, 10))
    en_body_plane   = np.zeros((10, 10))

    food_x = int(np_data[0]) - 1
    food_y = int(np_data[1]) - 1
 
    food_plane[food_x][food_y] = 1

    my_head_x = int(np_data[4]) - 1
    my_head_y = int(np_data[5]) - 1

    # set my head position to be equal to our snakes health
    if my_head_x < 10 and my_head_y < 10:
        my_head_plane[my_head_x][my_head_y] = np_data[2]

    # set my body positions
    for b in range(6, 24, 2):
        b_x = int(np_data[b]) - 1
        b_y = int(np_data[b + 1]) - 1

        if b_x < 0 or b_y < 0:
            continue

        my_body_plane[b_x][b_y] = 1
            

    en_head_x = int(np_data[24]) - 1
    en_head_y = int(np_data[25]) - 1

    # set enemy head position to be equal to the enemy health
    if en_head_x < 10 or en_head_y < 10:
        en_head_plane[en_head_x][en_head_y] = np_data[3]

    # set enemy body positions
    for b in range(24, 44, 2):
        b_x = int(np_data[b]) - 1
        b_y = int(np_data[b + 1]) - 1

        if b_x < 0 or b_y < 0:
            continue

        en_body_plane[b_x][b_y] = 1
    state = np.array([food_plane, my_head_plane, my_body_plane, en_head_plane, en_body_plane])
    print(state)
    return state

def json_obj_to_conv_input(json_obj):
    food_plane      = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    my_head_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    my_body_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    en_head_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))
    en_body_plane   = np.zeros((json_obj['board']['width'], json_obj['board']['height']))

    for food in json_obj['board']['food']:
        food_plane[food['y']][food['x']] = 1

    my_id = json_obj['you']['id']

    for snake in json_obj['board']['snakes']:
        if snake['id'] is not my_id:
            en_head_plane[snake['body'][0]['y']][snake['body'][0]['x']] = snake['health']
            for body in snake['body']:
                en_body_plane[body['y']][body['x']] = 1
        else:
            my_head_plane[snake['body'][0]['y']][snake['body'][0]['x']] = snake['health']
            for body in snake['body']:
                my_body_plane[body['y']][body['x']] = 1

    conv_input = np.array([food_plane, my_head_plane, my_body_plane, en_head_plane, en_body_plane])
    return conv_input

class ConvNet(nn.Module):
    def __init__(self, board_width, board_height):
        super(ConvNet, self).__init__()

        cuda0 = torch.device('cuda:0')

        self.conv1 = self.conv3x3(5, 10)
        self.conv2 = self.conv3x3(10, 20)
        self.fc1 = nn.Linear(20 * 10 * 10, 100)
        self.fc2 = nn.Linear(100, 4)

        # Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 0.001

        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor().to(cuda0))
        self.reward_episode = []

        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        return

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

    def conv3x3(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)

class ConvAI():
    def __init__(self, board_width, board_height):
        super(ConvAI, self).__init__()
        
        self.cuda0 = torch.device('cuda:0')
        self.net = ConvNet(board_width, board_height).to(self.cuda0)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.net.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

        self.episode = 0
        self.rewards = []
        self.policy_loss = []

        print(self.net)

        return

    def run_ai(self, input, training):
        net_input =  torch.from_numpy(json_obj_to_conv_input(input)).unsqueeze(0).to(self.cuda0).float()
        out = self.net(Variable(net_input))
        c = Categorical(out)
        action = c.sample()

        if training:
            # Add log probability of our chosen action to our history    
            if self.net.policy_history.dim() != 0:
                self.net.policy_history = torch.cat([self.net.policy_history, c.log_prob(action).to(self.cuda0)])
            else:
                self.net.policy_history = (c.log_prob(action)).to(self.cuda0)

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
        self.net.reward_episode.append(reward)
        return

    def update_policy(self):
        R = 0
        rewards = []
        
        # Discount future rewards back to the present using gamma
        for r in self.net.reward_episode[::-1]:
            R = r + self.net.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards).to(self.cuda0)
        rewards = rewards - rewards.mean()
        if rewards.std() > 0:
            rewards /= rewards.std()

        # Calculate loss
        loss = (torch.sum(torch.mul(self.net.policy_history, Variable(rewards)).to(self.cuda0).mul(-1) , -1))

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and inialize episode history counters
        self.net.loss_history.append(loss.data[0])
        self.net.reward_history.append(np.sum(self.net.reward_episode))
        self.net.policy_history = Variable(torch.Tensor()).to(self.cuda0)
        self.net.reward_episode = []

        self.episode += 1

        if self.episode % 100 == 0:
            episode_avg = self.net.loss_history[-100:]
            episode_avg = sum(episode_avg) / len(episode_avg)
            print('Trainer: episode ' + str(self.episode) + ' ' + str(loss) + ' ' + str(episode_avg))

        return

    def load(self):
        return


if __name__ == '__main__':
   run()