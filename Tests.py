import GridWorlds
import time
import numpy as np


env = GridWorlds.GridWorld(seed=1)

board = env.board
env.board[env.x,env.y] = env.board[env.x,env.y] -1
print(env.board)
env.board[1,1] = env.board[9,9] + 1
env.render()
time.sleep(9)