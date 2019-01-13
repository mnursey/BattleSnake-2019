import pygame
import math
import _global

class GameView:

    windowWidth = 800
    windowHeight = 600
    boardSizeX = 0
    boardSizeY = 0
    boardXOffset = 20
    boardYOffset = 20
    pixelPerSquare = 20
    squareOffset = 4

    backgroundColor = (180, 180, 180)
    defaultSquareColor = (200, 200, 200)
    defaultSnakeColor = (60, 60, 140)
    mySnakeColor = (60, 140, 60)
    ASnakeColor = (40, 160, 40)
    BSnakeColor = (140, 140, 60)
    CSnakeColor = (140, 60, 60)
    myDeadSnakeColor = (140, 140, 60)
    foodColor = (220, 60, 60)
    headColorOffset = -40
    health_bar_width = 200
    health_bar_height = 20
    health_bar_pos_x = 200
    health_bar_pos_y = 20

    def __init__(self):

        print("Opening Game View Window")

        self._running = True

        self._window = pygame.display.set_mode( (self.windowWidth, self.windowHeight) )
        pygame.display.set_caption("Game Viewer")

        self._snakes = None
        self._mySnake = None
        self._food = None
        self._grid = [[self.defaultSquareColor for y in range(self.boardSizeY)] for x in range(self.boardSizeX)]

        #self.draw()

        #self.update_pps()

        #self.update()
        
    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    _global.view_mode = 0
                if event.key == pygame.K_RIGHT:
                    _global.view_mode = 1
                if event.key == pygame.K_UP:
                    _global.view_mode = 2
                if event.key == pygame.K_DOWN:
                    _global.view_mode = 3

                if event.key == pygame.K_g:
                    if _global.enable_graph == 0:
                        _global.enable_graph = 1
                    else:
                        _global.enable_graph = 3

        self._window.fill(self.backgroundColor)
        self.draw()
        pygame.display.update()

    # pps (pixel per square)
    def update_pps(self): 
        self.pixelPerSquare = math.floor((min((self.windowHeight, self.windowWidth)) - max((self.boardXOffset, self.boardSizeY))) / (max((self.boardSizeX, self.boardSizeY)) + self.squareOffset))

    def draw(self):
        for x in range(self.boardSizeX):
            for y in range(self.boardSizeY):
                rect = pygame.Rect(self.boardSizeX + x * self.squareOffset + x * self.pixelPerSquare, self.boardYOffset + y * self.squareOffset + y * self.pixelPerSquare, self.pixelPerSquare, self.pixelPerSquare)
                pygame.draw.rect(self._window, self._grid[x][y], rect)

        if self._snakes is not None:
            
            for _, snake in enumerate(self._snakes):
                prevRect = None
                drawnHead = False
                if snake['health'] <= 0:
                    continue
                snakeColor = self.defaultSnakeColor

                if snake['id'] == self._mySnake['id']:
                    snakeColor = self.mySnakeColor

                if snake['id'] == 'A':
                    snakeColor = self.ASnakeColor

                if snake['id'] == 'B':
                    snakeColor = self.BSnakeColor

                if snake['id'] == 'C':
                    snakeColor = self.CSnakeColor

                health_bar_bg_rect = pygame.Rect(self.health_bar_pos_x, self.health_bar_pos_y + self.health_bar_height * _ * 1.5, self.health_bar_width, self.health_bar_height)
                health_bar_rect = pygame.Rect(self.health_bar_pos_x, self.health_bar_pos_y + self.health_bar_height * _ * 1.5, self.health_bar_width * snake['health'] / 100, self.health_bar_height)
                pygame.draw.rect(self._window, self.defaultSquareColor, health_bar_bg_rect)
                pygame.draw.rect(self._window, snakeColor, health_bar_rect)

                for body in snake['body']:
                    rect = pygame.Rect(self.boardSizeX + body['x'] * self.squareOffset + body['x'] * self.pixelPerSquare, self.boardYOffset + body['y'] * self.squareOffset + body['y'] * self.pixelPerSquare, self.pixelPerSquare, self.pixelPerSquare)
                    if prevRect is not None:
                        unionRect = rect.union(prevRect)
                        pygame.draw.rect(self._window, snakeColor, unionRect)
                        if not drawnHead:
                            pygame.draw.rect(self._window, (snakeColor[0] + self.headColorOffset, snakeColor[1] + self.headColorOffset, snakeColor[2] + self.headColorOffset), prevRect)
                            drawnHead = True
                    prevRect = rect

        if self._food is not None:
            for food in self._food:
                rect = pygame.Rect(self.boardSizeX + food['x'] * self.squareOffset + food['x'] * self.pixelPerSquare, self.boardYOffset + food['y'] * self.squareOffset + food['y'] * self.pixelPerSquare, self.pixelPerSquare, self.pixelPerSquare)
                pygame.draw.circle(self._window, self.foodColor, rect.center, math.floor(self.pixelPerSquare / 2) - 2)

    def update_grid(self, json):
        # handle json here... 

        if json != None and json != "":

            self._snakes = json['board']['snakes']
            self._mySnake = json['you']
            self._food = json['board']['food']

            self.boardSizeX = json['board']['width']
            self.boardSizeY = json['board']['height']

            self._grid = [[self.defaultSquareColor for y in range(self.boardSizeY)] for x in range(self.boardSizeX)]

            #self.update_pps()

    def finalize(self):
        pygame.quit()
