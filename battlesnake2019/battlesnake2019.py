import os
import _global
import time
import _thread
import server
import gameview as gv
import loggerserver
import data_formatter
import engine
import json
import sim_game
import ml_trainer_torch

running = False

enable_snake_server = False
enable_game_logging = False
snake_server_port   = '10013'

enable_game_viewer  = True

enable_log_server   = False
log_server_port     = '10010'


run_formatter_on_start = False
formatter_data_loc = './log_data/1v1_10by10_1f_2018snake'

run_ai_trainer_on_start = False

enable_sim_game = True

def Run():

    if run_formatter_on_start:
        data_formatter.simple_setup_test_one(formatter_data_loc)

    if run_ai_trainer_on_start:
        ml_trainer_torch.run()

    if enable_game_viewer:
        _thread.start_new_thread(RunBoard, ("Thread-Board",))

    if enable_snake_server:
        _thread.start_new_thread(RunServer, ("Thread-Game-Server",))

    if enable_log_server:
        _thread.start_new_thread(RunLogServer, ("Thread-Log-Server",))

    if enable_sim_game:
        _thread.start_new_thread(RunSimGame, ("Thread-Sim-Game",))

    return

def RunServer(thread_name):
    server.start(snake_server_port, enable_game_logging)

def RunBoard(thread_name):

    gameView = gv.GameView()

    while(gameView._running and running):
        gameView.update_grid(_global.board_json_list)
        gameView.update()
        time.sleep(0.30)

    gameView.finalize()

def RunLogServer(thread_name):
    loggerserver.start(log_server_port)

def RunSimGame(thread_name):
    sim_game.run()

if __name__ == '__main__':
    running = True
    Run()
