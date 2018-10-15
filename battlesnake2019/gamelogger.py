import json

game_dict = {}

def log(input, move):
    global json
    global game_dict

    game_id = input['game']['id']
    my_id = input['you']['id']

    if game_id not in game_dict:
        game_dict[game_id] = {}
        
        if my_id not in game_dict[game_id]:
            game_dict[game_id][my_id] = []

    turn = {'move' : move,'input' : input}

    game_dict[game_id][my_id].append(turn)

    return

def send_log(game_id, snake_id, log_collection):

    if game_id in game_dict:
        if snake_id in game_dict[game_id]:

            log_json = json.dumps({'collection' : log_collection, 'data' : game_dict[game_id][snake_id]}, indent=4)
            del game_dict[game_id][snake_id]
            # send json


    return