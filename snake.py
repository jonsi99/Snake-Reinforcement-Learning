import numpy as np
import random as rd
import os
from time import sleep

os.system('')

WIDTH = 21
PAD = 2

left  = {( 0, 1):(-1, 0),
         (-1, 0):( 0,-1),
         ( 0,-1):( 1, 0),
         ( 1, 0):( 0, 1)}

right = {( 0, 1):( 1, 0),
         ( 1, 0):( 0,-1),
         ( 0,-1):(-1, 0),
         (-1, 0):( 0, 1)}

NONE = 0
FOOD = 1
SNAKE = 2
WALL = 3

tiles = {NONE:'  ', FOOD:'<>', SNAKE:'[]', WALL:'##'}

class Snake:

    def __init__(self):
        self.len = 3
        self.body = np.zeros((WIDTH**2, 2), dtype=np.int8)
        for i in range(self.len):
            self.body[i] = [PAD+WIDTH//2, PAD+1+i]
        self.h = self.len - 1
        self.dir = (0,1)
        self.ate = False

    def hnext(self):
        return (self.h + 1) % self.len

    def head(self):
        return self.body[self.h]

    def tail(self):
        return self.body[self.hnext()]

    def turn(self, action):
        if action == 1: self.dir = left[self.dir]
        elif action == 2: self.dir = right[self.dir]

    def move(self, action):
        neck = self.head().copy()
        self.h = self.hnext()
        if self.ate:
            self.body[self.h+1:self.len+1] = self.body[self.h:self.len]
            self.len += 1
            self.ate = False
        self.turn(action)
        self.body[self.h] = neck + self.dir

class Game(Snake):

    def __init__(self):
        Snake.__init__(self)
        self.over = False
        self.grid = np.full((WIDTH+2*PAD, WIDTH+2*PAD), WALL, dtype=np.int8)
        self.grid[PAD:-PAD, PAD:-PAD] = NONE
        for i in range(self.len):
            self.gridloc(self.body[i], assign=SNAKE)
        self.food = (0,0)
        self.feed()
        
    def gridloc(self, idx, assign=None):
        if assign != None:
            self.grid[idx[0], idx[1]] = assign
        return self.grid[idx[0], idx[1]]

    def feed(self):
        while self.gridloc(self.food) > NONE:
            self.food = (rd.randrange(PAD,PAD+WIDTH), rd.randrange(PAD,PAD+WIDTH))
        self.gridloc(self.food, assign=FOOD)

    def act(self, action):
        if not self.ate:
            self.gridloc(self.tail(), assign=NONE)
        self.move(action)
        if self.gridloc(self.head()) >= SNAKE:
            self.over = True
        elif self.gridloc(self.head()) == FOOD:
            self.ate = True
            self.feed()
        self.gridloc(self.head(), assign=SNAKE)

    def print(self):
        print('\33[H', end='\r')
        for i in range(WIDTH + 2*PAD):
            for j in range(WIDTH + 2*PAD):
                print(tiles[self.grid[i,j]], end='')
            print()
        print(self.len)

    def compass(self):
        d = self.food - self.head()
        if d[0] < 0:
            if d[1] < 0: return 0
            if d[1] > 0: return 1
            return 2
        if d[0] > 0:
            if d[1] < 0: return 3
            if d[1] > 0: return 4
            return 5
        if d[1] < 0: return 6
        return 7

    def state(self):
        state = abs(self.dir[0])
        state += 1 + self.dir[0] + self.dir[1]
        if self.gridloc(self.head() + self.dir) >= SNAKE: state += 4
        if self.gridloc(self.head() + left[self.dir]) >= SNAKE: state += 8
        if self.gridloc(self.head() + right[self.dir]) >= SNAKE: state += 16
        state += 32 * self.compass()
        return state

    def reward(self):
        return self.ate - self.over

class AI:

    def __init__(self, seed=None, epsilon=0.1, alpha=0.1, gamma=0.9, iters=10000):
        rd.seed(seed)
        self.Q = np.zeros((256,3))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.train(iters)

    def update(self, s, a, r, ns):
        diff = r + self.gamma * max(self.Q[ns]) - self.Q[s,a]
        self.Q[s,a] += self.alpha * diff

    def learn(self):
        game = Game()
        s = game.state()
        while not game.over:
            if rd.random() < self.epsilon: a = rd.randrange(3)
            else: a = np.argmax(self.Q[s])
            game.act(a)
            ns = game.state()
            self.update(s, a, game.reward(), ns)
            s = ns

    def train(self, iters):
        print('TRAINING')
        for i in range(1, iters+1):
            self.learn()
            print(str(i) + '/' + str(iters), end='\r')
        print()

    def run(self, seed=None, delay=0.01):
        print('\33[2J')
        rd.seed(seed)
        game = Game()
        game.print()
        while not game.over:
            sleep(delay)
            game.act(np.argmax(self.Q[game.state()]))
            game.print()
