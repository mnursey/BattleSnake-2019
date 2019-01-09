import random
import copy
import pickle
import engine
import _global
import snake_random

gs = 0

# one in n chance of mutation
mutation_chance = 100 

# heuristic hyperparameters
h = {
    'dead' : 10,
    'alive' : 100,
    'last_alive': 2000,
    'turn_alive': 25,
    'full' : 200,
    'not_next_to_wall' : 25,
    'points_for_free_space': 75
}

print(h)


def run_ai(K, I, N, G, state):
    global gs
    gs = 0
    preds = {}

    state['stats'] = {}

    for snake in state['board']['snakes']:

        state['stats'][snake['id']] = {'turns_alive': 0}

        preds[snake['id']] = []
        for g in range(G):
            sequence = []
            for n in range(N):
                sequence.append(function_lookup[random.randint(0, len(function_lookup) - 1)])
            preds[snake['id']].append(sequence)

    for i in range(I):
        ratings = {}
        for snake in state['board']['snakes']:
            ratings[snake['id']] = []
            for s in range(len(preds[snake['id']])):
                temp_state = pickle.loads(pickle.dumps(state, -1))
                temp_state = simulate(temp_state, s, snake['id'], preds)
                ratings[snake['id']].append(rate(temp_state, state, snake['id'], N))
        improve_preds(preds, ratings)

    # find best move
    ratings = {}
    for snake in state['board']['snakes']:
        ratings[snake['id']] = []
        for s in range(len(preds[snake['id']])):
            temp_state = pickle.loads(pickle.dumps(state, -1))
            temp_state = simulate(temp_state, s, snake['id'], preds)
            ratings[snake['id']].append(rate(temp_state, state, snake['id'], N))

    best = 0
    best_index = 0
    for _, rating in enumerate(ratings[state['you']['id']]):
        if best < rating:
            best = rating
            best_index = _

    #print(gs)

    return preds[state['you']['id']][best_index][0]

def simulate(state, s, id, preds):
    global gs
    gs += 1

    en_seq = 0

    for i in range(len(preds[id][s])):
        # add moves
        moves = []
        #print('viewing sim game')
        for snake in state['board']['snakes']:
            if snake['id'] == id:
                moves.append((preds[id][s][i], id))
                #print(preds[id][s])
            else:
                moves.append((preds[snake['id']][en_seq][i], snake['id']))
                #print(preds[snake['id']][0])
        
        # sim moves
        if len(moves) > 1:
            state = engine.Run(state, moves, True) 
            #_global.board_json_list = state

            for snake in state['board']['snakes']:
                state['stats'][snake['id']]['turns_alive'] += 1
        else:
            break
    # return final state

    return state

def rate(state, prev_state, id, N):
    score = 0
    turns_played = state['turn'] - prev_state['turn']
    #turns alive
    score += h['turn_alive'] * state['stats'][id]['turns_alive']
    grid = None
    # dead or alive
    found = False
    for snake in state['board']['snakes']:
        if snake['id'] == id:
            found = True
            if snake['health'] > 20:
                score += h['full']

            score += h['alive']
            if len(state['board']['snakes']) == 1:
                score += h['last_alive'] * (N + 1 - turns_played)

            head = snake['body'][0]
            if head['x'] != 0 and head['y'] != 0 and head['x'] != state['board']['width'] - 1 and head['y'] != state['board']['height'] - 1:
                score += h['not_next_to_wall']
            
            grid = snake_random.generate_grid(state)
            state['you'] = snake
            adj = snake_random.get_free_moves(state, grid)
            score += len(adj) * h['points_for_free_space']
            break

    if not found:
        score += h['dead']

    return score

def chase_tail(state, grid):
    return

def get_food(state, grid):
    return

def improve_preds(preds, ratings):

    for i, snake_preds in enumerate(preds):
        score_sum = sum(ratings[snake_preds])

        # repopulate            
        children = []
        for g in range(len(preds[snake_preds])):

            # select parent A
            temp_value = random.randint(0, score_sum)
            for b, rating in enumerate(ratings[snake_preds]):
                temp_value -= rating
                if temp_value <= 0:
                    parent_a = preds[snake_preds][b]
                    break

            # select parent B
            temp_value = random.randint(0, score_sum)
            for b, rating in enumerate(ratings[snake_preds]):
                temp_value -= rating
                if temp_value <= 0:
                    parent_b = preds[snake_preds][b]
                    break

            # create child
            child = []
            for l in range(len(parent_a)):

                if random.randint(1, mutation_chance) == 1:
                    child.append(random.randint(0, len(function_lookup) - 1))
                else:
                    if random.randint(0, 1) == 0:
                        child.append(parent_a[l])
                    else:
                        child.append(parent_b[l])

            children.append(child)

        preds[snake_preds] = children

    return

function_lookup = [
   'up',
   'down',
   'left',
   'right'
   ]