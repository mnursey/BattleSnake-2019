import engine
import os
import _global
import matplotlib
import matplotlib.pyplot as plt
import json
import time 
import snake2018
import snake_random
import ml_trainer_torch
import copy
import random
import sage_serpent
import pickle
import ff_snake
import _thread
import numpy as np

graph_update = 100

graphs_plots_a = [[0],[0]]
graphs_plots_g = [[0],[0]]
graphs_plots_a_loss = [[0],[0]]
graphs_plots_a_reward = [[0],[0]]
a_win = 0
a_sum_of_scores = 0
game_number = 0
sum_of_game_length = 0
max_turns = 50000

batch_size = 100

#board_sizes = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 14, 15, 16, 17, 18, 19, 19]
board_sizes = [7, 7, 7, 8, 9, 10, 11, 11, 11, 12, 13]

h_index = 0
h = [{
    'win' : 0.2,
    'loss': -1.0,
    'tie': -0.9,
    'ate': 1.0,
    'initial': 0.0,
    'greedy_attack': 0.0,
    'retreat' : 0.0,
    }
]

print(h)

def RunGraph():
    if  _global.enable_graph == 0:
        return

    if  _global.enable_graph == 1:
        plt.dpi = 200
        plt.ion()
        plt.show()
        _global.enable_graph = 2


    if game_number % graph_update == 0 and _global.enable_graph == 2:
        plt.clf()
        ax = plt.subplot(3, 1, 1)
        ax.set_ylim([-5, 100])
        plt.axhline(0, color='black')
        a_plot_w_x = graphs_plots_a[0]
        a_plot_w_y = graphs_plots_a[1]
        g_plot_w_x = graphs_plots_g[0]
        g_plot_w_y = graphs_plots_g[1]
        plt.plot(a_plot_w_x, a_plot_w_y, 'r-')
        plt.plot(g_plot_w_x, g_plot_w_y, 'g-')
        plt.ylabel('Wins')
        plt.title('ML Graphs')

        plt.subplot(3, 1, 2, sharex=ax)
        plt.axhline(0, color='black')
        plt.plot(graphs_plots_a_loss[0], graphs_plots_a_loss[1], 'r-')
        plt.ylabel('Average Loss')

        plt.subplot(3, 1, 3, sharex=ax)
        plt.axhline(0, color='black')
        plt.plot(graphs_plots_a_reward[0], graphs_plots_a_reward[1], 'r-')
        plt.ylabel('Average Reward')
        plt.xlabel('Periods')

    plt.draw()
    plt.pause(0.0001)

    if _global.enable_graph == 3:
        plt.close()
        _global.enable_graph = 0

def run():

    global a_win
    global a_sum_of_scores
    global a_sum_of_rewards
    global game_number
    global sum_of_game_length
    global max_turns
    global batch_size

    global graphs_plots_a
    global graphs_plots_g
    global graphs_plots_a_loss
    global graphs_plots_a_reward
    global graphs_plots_b_reward 

    global h
    global h_index

    testing = False
    original_state = load_initial_state()

    path = './models/specialmodels/lIGbp5000.pth'
    ff_a = ff_snake.Policy(batch_size, False)
    ff_b = ff_snake.Policy(batch_size, True)   
    
    while True:
        game_number += 1
        
        # For Batch Size Generate Initial State
        states = [pickle.loads(pickle.dumps(original_state, -1)) for i in range(batch_size)]
        states_status = [0 for i in range(batch_size)]
        states_done = 0

        # Add variation to states
        for state in states:
            # size
            size = random.randint(0, len(board_sizes) - 1)
            size = board_sizes[size]

            state['board']['width'] = size
            state['board']['height'] = size

            # positions
            positions = [(1, 1), (size - 2, size - 2)]

            random.shuffle(positions)

            for p, snake in enumerate(state['board']['snakes']):
                for body in snake['body']:
                    body['x'] = positions[p][0]
                    body['y'] = positions[p][1]
            
            # food
            r_food = random.randint(0, size - 5)

            for i in r_food:
                state['board']['food'].append({'x': 0, 'y': 0})

            for food in state['board']['food']:
                while True:
                    x = random.randint(0, state['board']['width'] - 1)
                    y = random.randint(0, state['board']['height'] - 1)
                    if (x, y) not in positions:
                        positions.append((x, y))
                        food['x'] = x
                        food['y'] = y
                        break


        while states_done != batch_size:
            c = 0
            unfinished_states = []
            for i in range(batch_size):
                if states_status[i] == 0:
                    unfinished_states.append(i)

            # Get input 
            # grid = snake_random.generate_grid(state)

            a_input = np.array()
            b_input = np.array()
            for i in range(unfinished_states):
                state = states[unfinished_states[i]]

                for snake in state['board']['snakes']:
                    if snake['id'] == 'A':
                        state['you'] = snake

                    if snake['id'] == 'B':
                        state['you'] = snake


            # Get moves

            # Update States

            # Score states

        # Update ML

        RunGraph()
    return

def load_initial_state():
    # initial_game_food.json
    # initial_game.json
    file = open('./initial_game.json', 'r')
    json_ = file.read()
    obj_ = json.loads(json_)
    file.close()
    return obj_
