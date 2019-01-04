import engine
import os
import _global
import json
import time
import snake2018
import snake_random
import ml_trainer_torch
import copy
import random
import matplotlib.pyplot as plt
import sage_serpent
import pickle

enable_graph = False

def run():
  
    win = 0
    loss = 0
    bad_loss = 0
    hunger_loss = 0
    sum_of_scores = 0
    game_number = 0
    sum_of_game_length = 0
    max_turns = 1000
    batch_size = 500

    testing = False
    #pg_conv_agent = ml_trainer_torch.ConvAI(5, 5, batch_size)
    #pg_conv_agent.load('FBrbx385000')
    original_state = load_initial_state()

    graphs_plots_r = [[0],[0]]
    graphs_plots_b = [[0],[500]]
    graphs_plots_g = [[0],[0]]
    graphs_plots_y = [[0],[0]]
    graphs_plots_p = [[0],[0]]

    graph_update = 500

    if enable_graph:
        plt.ylabel('Wins over ' + str(graph_update) + ' games')
        plt.xlabel('Periods')
        plt.dpi = 200
        plt.axhline(0, color='black')
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.000001)

    while game_number < 10000:

        game_number += 1

        # new episode
        #pg_conv_agent.new_episode()

        done = False
        # copy original state
        state = pickle.loads(pickle.dumps(original_state, -1))

        positions = []

        # position snakes randomly
        for snake in state['board']['snakes']:
            while True:
                x = random.randint(0, state['board']['width'] - 1)
                y = random.randint(0, state['board']['height'] - 1)
                if (x, y) not in positions:
                    positions.append((x, y))
                    for body in snake['body']:
                        body['x'] = x
                        body['y'] = y
                    break

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

        #sage_serpent.run_ai(1, 1, 5, 5, state)
      
        while len(state['board']['snakes']) > 1 and not done:
            _global.board_json_list = state

            zero_health = False

            moves = []
            ai_surrounding_space = []
            grid = snake_random.generate_grid(state)
            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    state['you'] = snake
                    #my_move = pg_conv_agent.run_ai(state, testing)
                    my_move = snake_random.run_ai(state, grid)

                    moves.append((my_move, 'A'))
                    ai_surrounding_space = snake_random.get_free_moves(state, grid)
                    if snake['health'] <= 1:
                        zero_health = True
                if snake['id'] == 'B':
                    state['you'] = snake
                    moves.append((snake_random.run_corners_ai(state, grid), 'B'))

            state = engine.Run(state, moves) 

            found = False
            enemy_found = False
            ate = False
            reward = 0.0
            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    found = True
                    if snake['health'] == 99:
                        ate = True
                    
                if snake['id'] == 'B':
                    enemy_found = True
            
            if ate:
                reward =  0.0

            if found and not enemy_found:
                win += 1
                reward = 1.0
                   
                
            if not found or state['turn'] > max_turns:
                loss += 1
                if len(ai_surrounding_space) > 0 and not zero_health:
                    bad_loss += 1
                if zero_health:
                    hunger_loss += 1
                reward = -1.0
                done = True

            #pg_conv_agent.set_reward(reward)

        #if game_number % batch_size == 0 and not testing:
        #    sum_of_scores += pg_conv_agent.update_policy()

        sum_of_game_length += state['turn']

        if game_number % graph_update == 0:
            if loss == 0:
                loss = -1
            print('Sim: ' + str(win) + '/' + str(loss) + '/' + str(float(win)/float(loss + win)))

            # Update Graph
            if enable_graph:
                plt.axis([0, game_number / graph_update + game_number / graph_update * 0.1, -150, graph_update])
                graphs_plots_r[0].append(game_number / graph_update)
                graphs_plots_r[1].append(win)
                graphs_plots_b[0].append(game_number / graph_update)
                graphs_plots_b[1].append(bad_loss)
                graphs_plots_g[0].append(game_number / graph_update)
                graphs_plots_g[1].append(sum_of_scores * 100 / float(graph_update))
                graphs_plots_y[0].append(game_number / graph_update)
                graphs_plots_y[1].append(sum_of_game_length / float(graph_update))
                graphs_plots_p[0].append(game_number / graph_update)
                graphs_plots_p[1].append(hunger_loss)
                plt.plot(graphs_plots_r[0],graphs_plots_r[1], 'r-')
                plt.plot(graphs_plots_b[0],graphs_plots_b[1], 'b-')
                plt.plot(graphs_plots_g[0],graphs_plots_g[1], 'g-')
                plt.plot(graphs_plots_y[0],graphs_plots_y[1], 'y-')
                plt.plot(graphs_plots_p[0],graphs_plots_p[1], '-', color = 'purple')

                plt.draw()
                plt.pause(0.000001)

            loss = 0
            win = 0
            bad_loss = 0
            hunger_loss = 0
            sum_of_scores = 0
            sum_of_game_length = 0

    return

def load_initial_state():
    file = open('./initial_game.json', 'r')
    json_ = file.read()
    obj_ = json.loads(json_)
    file.close()
    return obj_
