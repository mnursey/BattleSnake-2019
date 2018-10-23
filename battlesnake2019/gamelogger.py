import json
import requests

server_addr = 'http://192.168.1.102:10010'

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
            # send json
            response = requests.post(server_addr + '/log', timeout = 100, json = {'collection' : log_collection, 'data' : game_dict[game_id][snake_id]})
          
            del game_dict[game_id][snake_id]

            print('Logger: sent game log to ' + server_addr)
            print('Logger: Log Server Response - ' + response.text)

    return