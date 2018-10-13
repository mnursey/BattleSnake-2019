import _global
import time
import _thread
import server
import gameview as gv

running = False

def Run():
     
    _thread.start_new_thread(RunBoard, ("Thread-Board",))
    _thread.start_new_thread(RunServer, ("Thread-Server",))
    return

def RunServer(thread_name):
    server.start('10013')

def RunBoard(thread_name):

    gameView = gv.GameView()

    while(gameView._running and running):
        gameView.update_grid(_global.board_json_list)
        gameView.update()
        time.sleep(0.30)

    gameView.finalize()


if __name__ == '__main__':
    running = True
    Run()
