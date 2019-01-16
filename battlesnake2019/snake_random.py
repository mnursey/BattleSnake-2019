import random

EMPTY = 0
WALL = 1
SNAKE = 2
FOOD = 3
TAIL = 4

OBSTACLES = [WALL, SNAKE, TAIL]

def generate_grid(state):
    grid = [[0 for col in range(state['board']['height'])] for row in range(state['board']['width'])]

    #for food in data['board']['food']:
        #grid[food['x']][food['y']] = FOOD

    for snake in state['board']['snakes']:
        for coord in snake['body']:
            grid[coord['x']][coord['y']] = SNAKE
        grid[snake['body'][-1]['x']][snake['body'][-1]['y']] = TAIL
    return grid

def get_free_moves(state, grid):
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

    return options

def get_adjacent_spaces(state, grid):
    options = []
    head = state['you']['body'][0]

    if head['y'] - 1 >= 0:
        options.append(grid[head['x']][head['y'] - 1])
    else:
        options.append(WALL)

    if head['y'] + 1 < state['board']['height']:
        options.append(grid[head['x']][head['y'] + 1])
    else:
        options.append(WALL)

    if head['x'] - 1 >= 0:
            options.append(grid[head['x'] - 1][head['y']])
    else:
        options.append(WALL)

    if head['x'] + 1 < state['board']['width']:
        options.append(grid[head['x'] + 1][head['y']])
    else:
        options.append(WALL)

    return options

def run_ai(state, grid = None):
    if grid is None:
        grid = generate_grid(state)

    options = get_free_moves(state, grid)

    move = 'up'
    if len(options) is 0:
        return move

    r = random.randint(0, len(options) - 1)
    
    return options[r]

def get_distance(x, y, a, b):
    disX = abs(x - a)
    disY = abs(y - b)
    return disX + disY

# move towards ab
def move_towards(x, y, a, b):

    move = None

    if x < a:
        move = 'right'
    if x > a: 
        move = 'left'
    if y < b:
        move = 'down'
    if y > b: 
        move = 'up'

    return move

# move towards ab
def move_towards_list(x, y, a, b):

    moves = []

    if x < a:
        moves.append('right')
    if x > a: 
        moves.append('left')
    if y < b:
        moves.append('down')
    if y > b: 
        moves.append('up')

    return moves

def run_corners_ai(state, grid = None):
    if grid is None:
        grid = generate_grid(state)

    you = state['you']

    adj_spaces = get_adjacent_spaces(state, grid)
    
    grid_width = state['board']['width']
    grid_height = state['board']['height']

    corners = [[0,0], [0, grid_height - 1], [grid_width - 1, 0], [grid_width - 1, grid_height - 1]]

    move = 'up'

    # get closest corner
    prev_closest = 999
    c = 0
    for i, corner in enumerate(corners):
        head_dis = abs(you['body'][0]['x'] - corner[0]) + abs(you['body'][0]['y'] - corner[1])
        #head_dis = get_distance(you['body'][0]['x'], you['body'][0]['y'], corner[0], corner[1])
        tail_dis = abs(you['body'][-1]['x'] - corner[0]) + abs(you['body'][-1]['y'] - corner[1])
        #tail_dis = get_distance(you['body'][-1]['x'], you['body'][-1]['y'], corner[0], corner[1])

        if tail_dis < head_dis or head_dis == 0:
            continue
        if head_dis < prev_closest:
            prev_closest = head_dis
            c = i
    
    move = move_towards(you['body'][0]['x'], you['body'][0]['y'], corners[c][0], corners[c][1])

    make_random_move = False

    if move == 'up':
        if adj_spaces[0] in OBSTACLES:
            make_random_move = True

    if move == 'down':
        if adj_spaces[1] in OBSTACLES:
            make_random_move = True

    if move == 'left':
        if adj_spaces[2] in OBSTACLES:
            make_random_move = True

    if move == 'right':
        if adj_spaces[3] in OBSTACLES:
            make_random_move = True

    if make_random_move:
        options = get_free_moves(state, grid)
        if len(options) > 0:
            r = random.randint(0, len(options) - 1)
            move = options[r]

    return move

def run_food_ai(state, grid = None):

    if grid is None:
        grid = generate_grid(state)

    you = state['you']

    adj_spaces = get_adjacent_spaces(state, grid)
    
    grid_width = state['board']['width']
    grid_height = state['board']['height']

    goals = []

    for food in state['board']['food']:
        goals.append([food['x'], food['y']])

    move = 'up'

    # get closest goal
    prev_closest = 999
    c = 0
    for i, g in enumerate(goals):
        head_dis = get_distance(you['body'][0]['x'], you['body'][0]['y'], g[0], g[1])

        if head_dis == 0:
            continue
        if head_dis < prev_closest:
            prev_closest = head_dis
            c = i
    
    if len(goals) > 0:
        move = move_towards(you['body'][0]['x'], you['body'][0]['y'], goals[c][0], goals[c][1])

    make_backup_move = False

    if move == 'up':
        if adj_spaces[0] in OBSTACLES:
            make_backup_move = True

    if move == 'down':
        if adj_spaces[1] in OBSTACLES:
            make_backup_move = True

    if move == 'left':
        if adj_spaces[2] in OBSTACLES:
            make_backup_move = True

    if move == 'right':
        if adj_spaces[3] in OBSTACLES:
            make_backup_move = True

    if make_backup_move:
        m = ['up', 'down', 'left', 'right']
        for i, adj in enumerate(adj_spaces):
            if adj not in OBSTACLES:
                move = m[i]
                break

    return move