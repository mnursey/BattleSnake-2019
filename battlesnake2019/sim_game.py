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

def run():
   
    pg_conv_agent = ml_trainer_torch.ConvAI(10, 10)
    original_state = load_initial_state()

    while True:
        # copy original state
        state = copy.deepcopy(original_state)

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

        _global.board_json_list = state

        while len(state['board']['snakes']) > 0:
            moves = []

            for snake in state['board']['snakes']:
                if snake['id'] == 'A':
                    state['you'] = snake
                    moves.append((pg_conv_agent.run_ai(state, True), 'A'))
                if snake['id'] == 'B':
                    state['you'] = snake
                    moves.append((snake_random.run_ai(state), 'B'))

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

            if not found:
                reward = -1.0
            else:
                if ate:
                    reward = 0.2
                else:
                    if not enemy_found:
                        reward = 0.0
                    else:
                        reward = 0.0

            pg_conv_agent.set_reward(reward)

            _global.board_json_list = state
    
        pg_conv_agent.update_policy()

    return

def load_initial_state():
    file = open('./initial_game.json', 'r')
    json_ = file.read()
    obj_ = json.loads(json_)
    file.close()
    return obj_
