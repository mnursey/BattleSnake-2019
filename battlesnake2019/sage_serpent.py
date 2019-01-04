import random
import copy
import pickle

def run_ai(K, I, N, G, state):
    preds = {}

    for snake in state['board']['snakes']:
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
            for s in preds[snake['id']]:
                temp_state = pickle.loads(pickle.dumps(state, -1))
                temp_state = simulate(temp_state, s, snake['id'], preds)
                ratings[snake['id']].append(rate(temp_state, snake['id']))
        improve_preds(preds, ratings)
    return

def simulate(state, s, id, preds):
    return

def rate(state, id):
    return 25

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
                if random.randint(0, 1) == 0:
                    child.append(parent_a[l])
                else:
                    child.append(parent_b[l])

            children.append(child)
        preds[snake_preds] = children

    return

function_lookup = [
   'chase_tail',
   'get_food'
   ]