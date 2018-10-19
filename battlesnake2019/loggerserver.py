import bottle
import os
import time

@bottle.post('/log')
def save_log():
    data = bottle.request.json
    print('Log Server: Received Log')
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
    application = bottle.default_app()
    bottle.BaseRequest.MEMFILE_MAX = 100000000
    bottle.run(application, server='paste', host=os.getenv('IP', '0.0.0.0'), port=os.getenv('PORT', port))


