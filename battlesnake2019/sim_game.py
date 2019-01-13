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

graphs_plots_r = [[0],[0]]
graphs_plots_b = [[0],[graph_update]]
graphs_plots_g = [[0],[0]]
graphs_plots_y = [[0],[0]]
graphs_plots_p = [[0],[0]]

win = 0
loss = 0
bad_loss = 0
hunger_loss = 0
sum_of_scores = 0
game_number = 0
sum_of_game_length = 0
max_turns = 25
size_turn_bonus = 50
batch_size = 100

I = 10
N = 6
G = 120

h = {
    'win' : 0.5,
    'loss': -1.0,
    'ate': 0.1,
    'initial': 0.0
    }

h1 = {
    'win' : 3.0,
    'loss': -2.0,
    'ate': 0.001,
    'initial': 0.0
    }

print(h)
print(h1)
#print("I:{} N:{} G:{}".format(I, N, G))

def RunGraph():
    if  _global.enable_graph == 0:
        return

    global graphs_plots_r 
    global graphs_plots_b 
    global graphs_plots_g 
    global graphs_plots_y 
    global graphs_plots_p 
    global graph_current_data

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
        plt.plot(graphs_plots_r[0],graphs_plots_r[1], 'r-')
        plt.plot(graphs_plots_b[0],graphs_plots_b[1], 'b-')
        plt.plot(graphs_plots_g[0],graphs_plots_g[1], 'g-')
        plt.plot(graphs_plots_y[0],graphs_plots_y[1], 'y-')
        plt.plot(graphs_plots_p[0],graphs_plots_p[1], '-', color = 'purple')
        plt.draw()
        plt.pause(0.0001)

    if _global.enable_graph == 3:
        plt.close()
        _global.enable_graph = 0

def run():

    global win
    global loss
    global bad_loss
    global hunger_loss
    global sum_of_scores
    global game_number
    global sum_of_game_length
    global max_turns
    global size_turn_bonus
    global batch_size

    global graphs_plots_r 
    global graphs_plots_b 
    global graphs_plots_g 
    global graphs_plots_y 
    global graphs_plots_p 
    global graph_current_data

    global h
    global h1

    testing = False
    #pg_conv_agent = ml_trainer_torch.ConvAI(5, 5, batch_size)
    #pg_conv_agent.load('FBrbx385000')
    original_state = load_initial_state()
    ff = ff_snake.Policy(original_state['board']['width'], original_state['board']['height'] , batch_size)      

    while True:
        game_number += 1

        # new episode
        #pg_conv_agent.new_episode()
        ff.new_episode()

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

        # position snakes randomly
        '''for snake in state['board']['snakes']:
            while True:
                x = random.randint(0, state['board']['width'] - 1)
                y = random.randint(0, state['board']['height'] - 1)
                if (x, y) not in positions:
                    positions.append((x, y))
                    for body in snake['body']:
                        body['x'] = x
                        body['y'] = y
                    break'''

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

            zero_health = False

            moves = []
            ai_surrounding_space = []
            grid = snake_random.generate_grid(state)
            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    state['you'] = snake

                    #my_move = pg_conv_agent.run_ai(state, testing)

                    '''t0 = time.clock()
                    my_move = sage_serpent.run_ai(1, 10, 6, 100, state)
                    t1 = time.clock()
                    total = t1-t0'''
                    #print("sage: " + str(total * 1000) + "ms")

                    my_move = ff.run_ai(state)

                    moves.append((my_move, 'A'))

                    ai_surrounding_space = snake_random.get_free_moves(state, grid)
                    if snake['health'] <= 1:
                        zero_health = True

                if snake['id'] == 'B':
                    state['you'] = snake
                    moves.append((snake_random.run_corners_ai(state), 'B'))

            state = engine.Run(state, moves) 

            found = False
            enemy_found = False
            ate = False
            reward = h['initial']
            length = 0
            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    found = True
                    length = len(snake['body'])
                    if snake['health'] == 100:
                        ate = True
                    
                if snake['id'] == 'B':
                    enemy_found = True
            
            if ate:
                reward += h['ate']

            if found and not enemy_found:
                #print('win')
                win += 1
                reward += h['win']
                done = True       

            if not found or state['turn'] > max_turns + size_turn_bonus * max(length - 3, 0):
                loss += 1
                #print('loss')
                if len(ai_surrounding_space) > 0 and not zero_health:
                    bad_loss += 1
                if zero_health:
                    hunger_loss += 1
                    #print('no health')
                reward = h['loss']
                done = True

            # set rewards
            #pg_conv_agent.set_reward(reward)
            ff.set_reward(reward)
            _global.board_json_list = state

        # update policy
        if game_number % batch_size == 0 and not testing:
            #sum_of_scores += pg_conv_agent.update_policy()
            sum_of_scores += ff.update_policy()

        sum_of_game_length += state['turn']


        if game_number % graph_update == 0:

            print('Sim: ' + str(win) + '/' + str(loss) + '/' + str(hunger_loss) + '/' + str(float(win)/float(loss + win)))    
            
            graphs_plots_r[0].append(game_number)
            graphs_plots_r[1].append(win)
            graphs_plots_b[0].append(game_number)
            graphs_plots_b[1].append(bad_loss)
            graphs_plots_g[0].append(game_number)
            graphs_plots_g[1].append((sum_of_scores / graph_update * 100))
            graphs_plots_y[0].append(game_number )
            graphs_plots_y[1].append(sum_of_game_length / graph_update)
            graphs_plots_p[0].append(game_number)
            graphs_plots_p[1].append(hunger_loss)

            if win > 60:
                h = h1
                print('changing scoring')

            loss = 0
            win = 0
            bad_loss = 0
            hunger_loss = 0
            sum_of_scores = 0
            sum_of_game_length = 0

        RunGraph()
    return

def load_initial_state():
    file = open('./initial_game.json', 'r')
    json_ = file.read()
    obj_ = json.loads(json_)
    file.close()
    return obj_
