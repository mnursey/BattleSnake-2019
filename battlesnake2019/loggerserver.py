import bottle
import os
import time
import json

data_folder_name = 'log_data'

@bottle.post('/log')
def save_log():
    global json

    data = bottle.request.json
    print('Log Server: Received Log')
    collection = data['collection']

    if not os.path.isdir(data_folder_name + '/' + collection):
        os.mkdir(data_folder_name + '/' + collection)

    snake_id = data['data'][0]['you']['id']

    text_data = json.dumps(data['data'])

    file = open(data_folder_name + '/' + collection + '/' + snake_id + '.json', 'w')
    file.write(text_data)
    file.close()

    return { 'LOGGED GAME' }

@bottle.get('/log')
def status():
    return{}

@bottle.get('/')
def status():
    return{
        "<!DOCKTYPE html><html><head><title>2019 Data Logger</title><style>p{color:red;}</style></head><body><p>BattleSnake 2019 Data Logging Server by Mitchell Nursey.</p></body></html>"
}

def start(port):
    # Expose WSGI app (so gunicorn can find it)

    if not os.path.isdir(data_folder_name):
        os.mkdir(data_folder_name)

    application = bottle.default_app()
    bottle.BaseRequest.MEMFILE_MAX = 100000000
    bottle.run(application, server='paste', host=os.getenv('IP', '0.0.0.0'), port=os.getenv('PORT', port))


