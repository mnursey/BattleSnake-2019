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
graphs_plots_b = [[0],[0]]
graphs_plots_g = [[0],[0]]
graphs_plots_a_loss = [[0],[0]]
graphs_plots_b_loss = [[0],[0]]
graphs_plots_a_reward = [[0],[0]]
graphs_plots_b_reward = [[0],[0]]
a_win = 0
b_win = 0
c_win = 0
a_sum_of_scores = 0
b_sum_of_scores = 0
a_sum_of_rewards = 0
b_sum_of_rewards = 0
game_number = 0
sum_of_game_length = 0
size_turn_bonus = 50
max_turns = 500000
batch_size = 100

n_updates = 0

#board_sizes = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 14, 15, 16, 17, 18, 19, 19]
board_sizes = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]

h_index = 2
h = [{
    'win' : 0.2,
    'loss': -1.0,
    'tie': -0.9,
    'ate': 0.5,
    'initial': 0.0,
    'greedy_attack': 0.0,
    'retreat' : 0.0
    },
     {
    'win' : 0.2,
    'loss': -1.0,
    'tie': -0.9,    
    'ate': 0.5,
    'initial': 0.0,
    'greedy_attack': 0.0,
    'retreat' : 0.0
    },
     {
    'win' : 1.0,
    'loss': -1.0,
    'tie': -0.9,    
    'ate': 0.1,
    'initial': 0.0,
    'greedy_attack': 0.0,
    'retreat' : 0.0
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
        b_plot_w_x = graphs_plots_b[0]
        b_plot_w_y = graphs_plots_b[1]
        g_plot_w_x = graphs_plots_g[0]
        g_plot_w_y = graphs_plots_g[1]
        plt.plot(a_plot_w_x, a_plot_w_y, 'r-')
        plt.plot(b_plot_w_x, b_plot_w_y, 'b-')
        plt.plot(g_plot_w_x, g_plot_w_y, 'g-')
        plt.ylabel('Wins')
        plt.title('ML Graphs')

        plt.subplot(3, 1, 2, sharex=ax)
        plt.axhline(0, color='black')
        plt.plot(graphs_plots_a_loss[0], graphs_plots_a_loss[1], 'r-')
        plt.plot(graphs_plots_b_loss[0], graphs_plots_b_loss[1], 'b-')
        plt.ylabel('Average Loss')

        plt.subplot(3, 1, 3, sharex=ax)
        plt.axhline(0, color='black')
        plt.plot(graphs_plots_a_reward[0], graphs_plots_a_reward[1], 'r-')
        plt.plot(graphs_plots_b_reward[0], graphs_plots_b_reward[1], 'b-')
        plt.ylabel('Average Reward')
        plt.xlabel('Periods')

    plt.draw()
    plt.pause(0.0001)

    if _global.enable_graph == 3:
        plt.close()
        _global.enable_graph = 0

def run():

    global a_win
    global b_win
    global c_win
    global a_sum_of_scores
    global b_sum_of_scores
    global a_sum_of_rewards
    global b_sum_of_rewards
    global game_number
    global sum_of_game_length
    global max_turns
    global batch_size

    global graphs_plots_a
    global graphs_plots_b 
    global graphs_plots_g
    global graphs_plots_a_loss
    global graphs_plots_b_loss 
    global graphs_plots_a_reward
    global graphs_plots_b_reward 

    global n_updates
    
    global h
    global h_index

    testing = False
    original_state = load_initial_state()

    path = './models/specialmodels/version_4_2.pth'
    ff_a = ff_snake.Policy(batch_size, training = True, path=path)
    #ff_a.reset_optimizer()                            
    ff_b = ff_snake.Policy(batch_size, training = False, path=path)   
    
    while True:
        game_number += 1
            
        # new episode
        ff_a.new_episode()

        done = False

        # copy original state
        state = pickle.loads(pickle.dumps(original_state, -1))


        size = random.randint(0, len(board_sizes) - 1)
        size = board_sizes[size]

        state['board']['width'] = size
        state['board']['height'] = size

        positions = [(1, 1), (size - 2, size - 2)]

        # add extra snakes
        if random.randint(0, 100) > 50:
            state['board']['snakes'].append({
            "id": "C",
            "name": "Snake C",
            "health": 100,
            "body": [
                {
                "x": 5,
                "y": 5
                },
                {
                "x": 5,
                "y": 5
                },
                {
                "x": 5,
                "y": 5
                }
            ]
            })

            positions.append((1, size - 2))

            if random.randint(0, 100) > 50:
                state['board']['snakes'].append({
                "id": "D",
                "name": "Snake D",
                "health": 100,
                "body": [
                    {
                    "x": 5,
                    "y": 5
                    },
                    {
                    "x": 5,
                    "y": 5
                    },
                    {
                    "x": 5,
                    "y": 5
                    }
                ]
                })

                positions.append((size - 2, 1))

        # Position snakes in corners
        ##(1, 1), (size - 2, size - 2), (1, size - 2)


        random.shuffle(positions)

        for p, snake in enumerate(state['board']['snakes']):
            for body in snake['body']:
                body['x'] = positions[p][0]
                body['y'] = positions[p][1]

        # food
        #r_food = random.randint(3, size - 4)
        r_food = random.randint(4, size)

        for i in range(r_food):
            state['board']['food'].append({'x': 0, 'y': 0})

        # position food randomly
        for food in state['board']['food']:
            while True:
                x = random.randint(0, state['board']['width'] - 1)
                y = random.randint(0, state['board']['height'] - 1)
                if (x, y) not in positions:
                    positions.append((x, y))
                    food['x'] = x
                    food['y'] = y
                    break
      
        while not done:

            if _global.view_mode == 1:
                time.sleep(0.5)

            if _global.view_mode == 3:
                continue
            
            if _global.view_mode == 2:
                _global.view_mode = 3

            moves = []

            snakeA = None
            snakeB = None
            snakeC = None
            snakeD = None
            a_move = None
            b_move = None
            c_move = None
            d_move = None

            #grid = snake_random.generate_grid(state)

            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    state['you'] = snake
                    snakeA = snake
                    a_move = ff_a.run_ai(state)
                    moves.append((a_move, 'A'))

                if snake['id'] == 'B':
                    state['you'] = snake
                    snakeB = snake

                    #b_move = snake_random.run_ai(state)
                    b_move = ff_b.run_ai_test(state)

                    moves.append((b_move, 'B'))

                if snake['id'] == 'C':
                    state['you'] = snake
                    snakeC = snake

                    c_move = ff_b.run_ai_test(state)
                    #c_move = snake_random.run_ai(state)
                    #c_move = snake_random.run_ai(state)
                    moves.append((c_move, 'C'))

                if snake['id'] == 'D':
                    state['you'] = snake
                    snakeD = snake

                    #d_move = snake_random.run_ai(state)
                    d_move = ff_b.run_ai_test(state)
                    #d_move = snake2018.run_ai(state)
                    moves.append((d_move, 'D'))

            state = engine.Run(state, moves) 

            a_found = False
            b_found = False
            c_found = False
            d_found = False
            a_ate = False
            b_ate = False
            c_ate = False
            d_ate = False

            a_reward = h[h_index]['initial']
            b_reward = h[h_index]['initial']
            a_length = 0
            b_length = 0

            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    a_found = True
                    a_length = len(snake['body'])
                    if snake['health'] == 100:
                        a_ate = True
                    
                if snake['id'] == 'B':
                    b_found = True
                    b_length = len(snake['body'])
                    if snake['health'] == 100:
                        b_ate = True    
                        
                if snake['id'] == 'C':
                    c_found = True
                    if snake['health'] == 100:
                        c_ate = True      

                if snake['id'] == 'D':
                    d_found = True
                    if snake['health'] == 100:
                        d_ate = True            
                        
            if a_ate:
                a_reward += h[h_index]['ate']

            if a_found and not b_found and not c_found and not d_found:
                a_win += 1
                a_reward += h[h_index]['win']
                done = True

            if not a_found:
                a_reward = h[h_index]['loss']
                done = True   

            if state['turn'] > max_turns or (not a_found and not b_found and not c_found and not d_found):
                a_reward = h[h_index]['tie']
                done = True

            # set rewards
            ff_a.set_reward(a_reward)
            a_sum_of_rewards += a_reward
            #b_sum_of_rewards += b_reward

            _global.board_json_list = state
            RunGraph()

        # update policy
        if game_number % batch_size == 0 and not testing:
            a_sum_of_scores += ff_a.update_policy()

        sum_of_game_length += state['turn']


            
        if game_number % graph_update == 0:

            print('Sim: ' + str(a_win) + '/' + str(n_updates))
            
            graphs_plots_a[0].append(game_number)
            graphs_plots_a[1].append(a_win)
            graphs_plots_b[0].append(game_number)
            graphs_plots_b[1].append(b_win)
            graphs_plots_g[0].append(game_number)
            graphs_plots_g[1].append(sum_of_game_length / graph_update)
            graphs_plots_a_loss[0].append(game_number)
            graphs_plots_a_loss[1].append(a_sum_of_scores / graph_update)
            graphs_plots_b_loss[0].append(game_number)
            graphs_plots_b_loss[1].append(b_sum_of_scores / graph_update)
            graphs_plots_a_reward[0].append(game_number)
            graphs_plots_a_reward[1].append(a_sum_of_rewards / graph_update)
            graphs_plots_b_reward[0].append(game_number)
            graphs_plots_b_reward[1].append(n_updates)
            
        if a_win > 70:
            path = ff_a.save()
            ff_b = ff_snake.Policy(batch_size, True, path=path)
            ff_a.reset_optimizer()
            n_updates += 1
            a_win = 0
            
            '''if n_updates > 2:
                h_index = 1'''
                
        if game_number % graph_update == 0:

            a_win = 0
            b_win = 0
            c_win = 0
            a_sum_of_scores = 0
            b_sum_of_scores = 0
            a_sum_of_rewards = 0
            b_sum_of_rewards = 0
            sum_of_game_length = 0

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
