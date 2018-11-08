import random

EMPTY = 0
WALL = 1
SNAKE = 2
FOOD = 3

def generate_grid(state):
    grid = [[0 for col in range(state['board']['height'])] for row in range(state['board']['width'])]

    #for food in data['board']['food']:
        #grid[food['x']][food['y']] = FOOD

    for snake in state['board']['snakes']:
        for coord in snake['body']:
            grid[coord['x']][coord['y']] = SNAKE

    return grid

def run_ai(state):
    grid = generate_grid(state)
    options = []
    head = state['you']['body'][0]

    if head['x'] + 1 < state['board']['width']:
        if(grid[head['x'] + 1][head['y']] is EMPTY):
            options.append('right')

    if head['x'] - 1 >= 0:
        if(grid[head['x'] - 1][head['y']] is EMPTY):
            options.append('left')

    if head['y'] - 1 >= 0:
        if(grid[head['x']][head['y'] - 1] is EMPTY):
            options.append('up')

    if head['y'] + 1 < state['board']['height']:
        if(grid[head['x']][head['y'] + 1] is EMPTY):
            options.append('down')

    move = 'up'
    if len(options) is 0:
        return move

    r = random.randint(0, len(options) - 1)
    
    return options[r]