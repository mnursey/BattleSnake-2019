import random 

def Run(state, moves):

    # move heads
    
    heads = []

    for snake in state['board']['snakes']:

        new_head = {'x' : snake['body'][0]['x'], 'y' : snake['body'][0]['y']}

        s_move = 'None'
        for move in moves: 
            if move[1] == snake['id']:
                s_move = move[0]
            else:
                continue

            if s_move is 'up':
                new_head['y'] -= 1
            if s_move is 'down':
                new_head['y'] += 1
            if s_move is 'left':
                new_head['x'] -= 1
            if s_move is 'right':
                new_head['x'] += 1

            break

        if new_head == snake['body'][0]:
            new_head['y'] -= 1

        snake['body'].insert(0, new_head)
        heads.append((snake['body'][0] , snake['id']))

    # check for head to head collisions
    for head_a in heads:
        for head_b in heads:
            
            # check if same head
            if head_a[1] == head_b[1]:
                continue

            if head_a[0] == head_b[0]:
                length_a =0
                length_b = 0

                for snake in state['board']['snakes']:
                    if snake['id'] == head_a[1]:
                        length_a = len(snake['body'])
                    if snake['id'] == head_b[1]:
                        length_b = len(snake['body'])

                if length_a < length_b:
                    # kill a
                    for i in range(len(state['board']['snakes'])):
                        if state['board']['snakes'][i]['id'] == head_a[1]:
                            state['board']['snakes'].pop(i)
                            break
                   
                if length_a > length_b:
                    # kill b
                    for i in range(len(state['board']['snakes'])):
                        if state['board']['snakes'][i]['id'] == head_b[1]:
                            state['board']['snakes'].pop(i)
                            break
                if length_a == length_b:
                    # kill both
                    for i in range(len(state['board']['snakes'])):
                        if state['board']['snakes'][i]['id'] == head_a[1]:
                            state['board']['snakes'].pop(i)
                            break

                    for i in range(len(state['board']['snakes'])):
                        if state['board']['snakes'][i]['id'] == head_b[1]:
                            state['board']['snakes'].pop(i)
                            break

    # eat food
    num_food_to_gen = 0
    for food in state['board']['food'][:]:
        for snake in state['board']['snakes']:
            if food['x'] == snake['body'][0]['x'] and food['y'] == snake['body'][0]['y']:
                # eat food
                snake['body'].append({'x' : snake['body'][-1]['x'], 'y' : snake['body'][-1]['y']})
                snake['health'] = 100
                state['board']['food'].remove(food)
                num_food_to_gen += 1
                break

    # gen food
    while num_food_to_gen > 0:
        for i in range(num_food_to_gen):
            x = random.randint(0, state['board']['width'] - 1)
            y = random.randint(0, state['board']['height'] - 1)

            collision = False
            for snake in state['board']['snakes']:
                if collision: break
                for body in snake['body']:
                    if body['x'] == x and body['y'] == y:
                        collision = True
                        break

            if not collision:
                num_food_to_gen -= 1
                state['board']['food'].append({'x' : x, 'y' : y})

    # remove tail
    for snake in state['board']['snakes']:
        snake['body'].pop(-1)

    for snake in state['board']['snakes'][:]:
        head = snake['body'][0]
        snake['health'] -= 1

        # check if snake starved
        if snake['health'] <= 0:
            # kill snake
            state['board']['snakes'].remove(snake)
            continue

        # check for head out of bounds

        if head['x'] < 0 or head['y'] < 0 or head['x'] >= state['board']['width'] or  head['y'] >= state['board']['height']: 
            # kill snake
            state['board']['snakes'].remove(snake)
            continue

        # check for collisions
        for other in state['board']['snakes'][:]:
            for i in range(len(other['body'])):
                if other['id'] == snake['id'] and i == 0:
                    continue
                if other['body'][i]['x'] == head['x'] and other['body'][i]['y'] == head['y']:
                    # kill snake
                    if snake in state['board']['snakes']:
                        state['board']['snakes'].remove(snake)

    state['turn'] += 1
    # return a state for each snake? with differnt you values?
    # state['you'] = None
    return state