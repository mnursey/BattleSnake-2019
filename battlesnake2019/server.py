import bottle
import os
import time
import _global
import snake2018
import gamelogger
import subprocess

log_collection = '1v1_10by10_1f_2018snake'
game_server_location = 'D:\Desktop\BattleSnake\engine_0.2.1_Windows_x86_64'
game_server_cmd = ['python', 'run.py']

@bottle.post('/start')
def start():
    data = bottle.request.json

    return {
        'color': 'blue',
        'secondary_color': 'red',
        'name': '256 bit snake'
    }

@bottle.post('/end')
def end():
    data = bottle.request.json

    # run new game on game server
    p = subprocess.run(game_server_cmd, cwd = game_server_location)

    gamelogger.log(data, 'None')
    gamelogger.send_log(data['game']['id'], data['you']['id'], log_collection)

    return {}

@bottle.post('/move')
def move():
    data = bottle.request.json
    start = time.time()
    move = 'left'
    move = snake2018.run_ai(data)
    end = time.time()
    print('Time to get AI move: ' + str((end - start) * 1000) + 'ms')
    _global.board_json_list = data

    gamelogger.log(data, move)

    return {
        'move': move
    }

@bottle.get('/')
def status():
    return{
        "<!DOCKTYPE html><html><head><title>2019 Snake</title><style>p{color:orange;}</style></head><body><p>BattleSnake 2019 by Mitchell Nursey.</p></body></html>"
}

def start(port):
    # Expose WSGI app (so gunicorn can find it)
    application = bottle.default_app()
    bottle.run(application, server='paste', host=os.getenv('IP', '0.0.0.0'), port=os.getenv('PORT', port))


