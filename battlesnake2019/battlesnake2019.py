import gameview as gv
import time
import json

gameView = gv.GameView()

jsonString = """
{
  "game": {
    "id": "game-id-string"
  },
  "turn": 0,
  "board": {
    "height": 15,
    "width": 25,
    "food": [{
      "x": 0,
      "y": 0
    }],
    "snakes": 
    [
        {
          "id": "123",
          "name": "MySnakeName",
          "health": 100,
          "body": [
        {
          "x": 5,
          "y": 6
        },
        {
          "x": 5,
          "y": 5
        },
        {
          "x": 6,
          "y": 5
        },
        {
          "x": 7,
          "y": 5
        },
        {
          "x": 7,
          "y": 8
        }]
        },
        {
          "id": "abc",
          "name": "SnakeName",
          "health": 27,
          "body": [
          {
            "x": 2,
            "y": 9
          },
          {
            "x": 3,
            "y": 9
          },
          {
            "x": 3,
            "y": 10
          },
          {
            "x": 2,
            "y": 10
          },
          {
            "x": 1,
            "y": 10
          },
          {
            "x": 0,
            "y": 10
          },
          {
            "x": 0,
            "y": 9
          },
          {
            "x": 0,
            "y": 8
          },{
            "x": 1,
            "y": 8
          },
          {
            "x": 1,
            "y": 9
          }]
        }
    ]
  },
  "you": {
    "id": "123",
    "name": "MySnakeName",
    "health": 100,
    "body": [{
      "x": 5,
      "y": 5
    },
    {
      "x": 6,
      "y": 5
    },
    {
      "x": 7,
      "y": 5
    },
    {
      "x": 7,
      "y": 8
    }]
  }
}
"""

json = json.loads(jsonString)

gameView.update_grid(json)

while(gameView._running):
    gameView.update()
    time.sleep(0.30)

gameView.finalize()
