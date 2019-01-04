"""import os
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
        return out"""
