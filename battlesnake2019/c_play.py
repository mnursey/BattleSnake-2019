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

a_win = 0
b_win = 0
a_sum_of_scores = 0
b_sum_of_scores = 0
game_number = 0
sum_of_game_length = 0
size_turn_bonus = 50
max_turns = 50
batch_size = 100

h_index = 0
h = [{
    'win' : 0.0,
    'loss': -0.2,
    'ate': 0.8,
    'initial': 0.0,
    'greedy_attack': 0.0
    },
    {
    'win' : 0.5,
    'loss': -1.0,
    'ate': 0.01,
    'initial': -0.01,
    'greedy_attack': 0.1
    }
]

print(h)

def RunGraph():
    if  _global.enable_graph == 0:
        return

    global graphs_plots_a
    global graphs_plots_b 

    if  _global.enable_graph == 1:
        plt.dpi = 200
        plt.ion()
        plt.show()
        _global.enable_graph = 2


    if game_number % graph_update == 0 and _global.enable_graph == 2:
        plt.clf()
        ax = plt.subplot(2, 1, 1)
        ax.set_ylim([-5, 100])
        plt.axhline(0, color='black')
        a_plot_w_x = graphs_plots_a[0]
        a_plot_w_y = graphs_plots_a[1]
        b_plot_w_x = graphs_plots_b[0]
        b_plot_w_y = graphs_plots_b[1]
        plt.plot(a_plot_w_x, a_plot_w_y, 'r-')
        plt.plot(b_plot_w_x, b_plot_w_y, 'b-')
        plt.ylabel('Wins over ' + str(graph_update) + ' games')
        plt.xlabel('Periods')
        plt.title('ML Graphs')

        plt.subplot(2, 1, 2, sharex=ax)
        plt.axhline(0, color='black')
        g_plot_w_x = graphs_plots_g[0]
        g_plot_w_y = graphs_plots_g[1]
        plt.plot(g_plot_w_x, g_plot_w_y, 'g-')
        plt.ylabel('Average Game Length over ' + str(graph_update) + ' games')
        plt.xlabel('Periods')

    plt.draw()
    plt.pause(0.0001)

    if _global.enable_graph == 3:
        plt.close()
        _global.enable_graph = 0

def run():

    global a_win
    global b_win
    global a_sum_of_scores
    global b_sum_of_scores
    global game_number
    global sum_of_game_length
    global max_turns
    global batch_size

    global graphs_plots_a
    global graphs_plots_b 
    global graphs_plots_g

    global h
    global h_index

    testing = False
    original_state = load_initial_state()

    ff_a = ff_snake.Policy(original_state['board']['width'], original_state['board']['height'] , batch_size)      
    ff_b = ff_snake.Policy(original_state['board']['width'], original_state['board']['height'] , batch_size)   
    
    while True:
        game_number += 1

        # new episode
        ff_a.new_episode()
        ff_b.new_episode()

        done = False

        # copy original state
        state = pickle.loads(pickle.dumps(original_state, -1))

        pos_flip = random.randint(0, 1)
        positions = [(1,1), (5,5)]

        if pos_flip == 0:
            snake = state['board']['snakes'][0]

            for body in snake['body']:
                body['x'] = positions[0][0]
                body['y'] = positions[0][1]

            if len(state['board']['snakes']) > 1:
                snake = state['board']['snakes'][1]
                for body in snake['body']:
                    body['x'] = positions[1][0]
                    body['y'] = positions[1][1]
        else:
            snake = state['board']['snakes'][0]
            for body in snake['body']:
                body['x'] = positions[1][0]
                body['y'] = positions[1][1]

            if len(state['board']['snakes']) > 1:
                snake = state['board']['snakes'][1]
                for body in snake['body']:
                    body['x'] = positions[0][0]
                    body['y'] = positions[0][1]

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
            a_move = None
            b_move = None
            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    state['you'] = snake
                    snakeA = snake
                    a_move = ff_a.run_ai(state)
                    moves.append((a_move, 'A'))

                if snake['id'] == 'B':
                    state['you'] = snake
                    snakeB = snake
                    b_move = ff_b.run_ai(state)

                    moves.append((b_move, 'B'))

            greedy_attack_moves = snake_random.move_towards_list(snakeA['body'][0]['x'], snakeA['body'][0]['y'], snakeB['body'][0]['x'], snakeB['body'][0]['y'])
            state = engine.Run(state, moves) 

            a_found = False
            b_found = False
            a_ate = False
            b_ate = False

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
                        
            if a_ate:
                a_reward += h[h_index]['ate']
            if b_ate:
                b_reward += h[h_index]['ate']

            if a_move in greedy_attack_moves:
                a_reward +=h [h_index]['greedy_attack']
            if b_move not in greedy_attack_moves:
                b_reward +=h [h_index]['greedy_attack']

            if a_found and not b_found:
                a_win += 1
                a_reward += h[h_index]['win']
                b_reward = h[h_index]['loss']
                done = True

            if b_found and not a_found:
                b_win += 1
                a_reward = h[h_index]['loss']
                b_reward += h[h_index]['win']
                done = True   


            if state['turn'] > max_turns + size_turn_bonus * max(max(a_length - 3, 0), max(b_length - 3, 0)) or (not a_found and not b_found):
                a_reward = h[h_index]['loss']
                b_reward = h[h_index]['loss']
                done = True

            # set rewards
            ff_a.set_reward(a_reward)
            ff_b.set_reward(b_reward)
            _global.board_json_list = state
            RunGraph()

        # update policy
        if game_number % batch_size == 0 and not testing:
            a_sum_of_scores += ff_a.update_policy()
            b_sum_of_scores += ff_b.update_policy()

        sum_of_game_length += state['turn']

        if game_number % graph_update == 0:

            print('Sim: ' + str(a_win) + '/' + str(b_win))
            
            graphs_plots_a[0].append(game_number)
            graphs_plots_a[1].append(a_win)
            graphs_plots_b[0].append(game_number)
            graphs_plots_b[1].append(b_win)
            graphs_plots_g[0].append(game_number)
            graphs_plots_g[1].append(sum_of_game_length / batch_size)

            if sum_of_game_length / batch_size > 50 and h_index == 0:
                h_index = 1
                print('changing scoring')

            a_win = 0
            b_win = 0
            a_sum_of_scores = 0
            b_sum_of_scores = 0
            sum_of_game_length = 0

        RunGraph()
    return

def load_initial_state():
    file = open('./initial_game.json', 'r')
    json_ = file.read()
    obj_ = json.loads(json_)
    file.close()
    return obj_
