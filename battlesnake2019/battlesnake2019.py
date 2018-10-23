import _global
import time
import _thread
import server
import gameview as gv
import loggerserver
import data_formatter

running = False

enable_snake_server = True
enable_game_viewer  = True
enable_log_server   = True
run_formatter_on_start = False
snake_server_port   = '10013'
log_server_port     = '10010'

def Run():
    
    if run_formatter_on_start:
        data_formatter.simple_setup_test_one('./log_data/1v1_10by10_1f_2018snake')
    if enable_game_viewer:
        _thread.start_new_thread(RunBoard, ("Thread-Board",))
    if enable_snake_server:
        _thread.start_new_thread(RunServer, ("Thread-Game-Server",))
    if enable_log_server:
        _thread.start_new_thread(RunLogServer, ("Thread-Log-Server",))
    return

def RunServer(thread_name):
    server.start(snake_server_port)

def RunBoard(thread_name):

    gameView = gv.GameView()

    while(gameView._running and running):
        gameView.update_grid(_global.board_json_list)
        gameView.update()
        time.sleep(0.30)

    gameView.finalize()

def RunLogServer(thread_name):
    loggerserver.start(log_server_port)

if __name__ == '__main__':
    running = True
    Run()
