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
max_turns = 500
batch_size = 100

h_index = 0
h = [{
    'win' : 0.001,
    'loss': -1.0,
    'ate': 0.8,
    'initial': 0.0
    },
    {
    'win' : 3.0,
    'loss': -2.0,
    'ate': 0.001,
    'initial': -0.001
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
        plt.ylabel('Wins over ' + str(graph_update) + ' games')
        plt.xlabel('Periods')
        plt.axhline(0, color='black')
        plt.ion()
        plt.show()
        _global.enable_graph = 2


    if game_number % graph_update == 0 and _global.enable_graph == 2:
        plt.axis([0, game_number + game_number * 0.1 , -5, graph_update])
        plt.plot(graphs_plots_a[0],graphs_plots_a[1], 'r-')
        plt.plot(graphs_plots_b[0],graphs_plots_b[1], 'b-')
        plt.plot(graphs_plots_g[0],graphs_plots_g[1], 'g-')
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

            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    state['you'] = snake
                    a_move = ff_a.run_ai(state)

                    moves.append((a_move, 'A'))

                if snake['id'] == 'B':
                    state['you'] = snake
                    b_move = ff_b.run_ai(state)

                    moves.append((b_move, 'B'))

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

            if a_found and not b_found:
                a_win += 1
                a_reward += h[h_index]['win']
                b_reward += h[h_index]['loss']
                done = True

            if b_found and not a_found:
                b_win += 1
                a_reward += h[h_index]['loss']
                b_reward += h[h_index]['win']
                done = True   

            if state['turn'] > max_turns or (not a_found and not b_found):
                a_reward += h[h_index]['loss']
                b_reward += h[h_index]['loss']
                done = True

            # set rewards
            ff_a.set_reward(a_reward)
            ff_b.set_reward(b_reward)
            _global.board_json_list = state

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

            if sum_of_game_length / batch_size > 75 and h_index == 0:
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
