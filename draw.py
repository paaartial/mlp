import pygame, sys, time

import numpy as np

from network import Network
from run import load_network, conv

from digits import numbers

from network import Network
from run import load_network

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
window_height = 800
window_width = 1440
block_size=20
grid_size=[28, 28]

grid_start=[int(0.04*window_width), int(0.13*window_height)]
num_start = [775, 350]

grid = [[255 for x in range(grid_size[0])] for y in range(grid_size[1])]

from run import kernel_size
grid_transform = np.zeros(shape = (1, grid_size[0], grid_size[1]))

guesser = load_network("net", new_name="guesser")

global SCREEN, CLOCK
pygame.init()
SCREEN = pygame.display.set_mode((window_width, window_height))
CLOCK = pygame.time.Clock()
SCREEN.fill(BLACK)


def drawGrid():
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            y_coord=y*block_size + grid_start[1]
            x_coord=x*block_size + grid_start[0]
            rect = pygame.Rect(x_coord, y_coord, block_size, block_size)
            pygame.draw.rect(SCREEN, (grid[x][y], grid[x][y], grid[x][y]), rect)

def drawNum(num):
    for y in range(len(num)):
        for x in range(len(num[y])):
            y_coord=y*5 + num_start[1]
            x_coord=x*5 + num_start[0]
            rect = pygame.Rect(x_coord, y_coord, 5, 5)
            pygame.draw.rect(SCREEN, ((1-num[y][x])*255, (1-num[y][x])*255, (1-num[y][x])*255), rect)

def vector_to_coord(pos):
    out_of_range = False
    x_coord = pos[0]-grid_start[0]  
    y_coord = pos[1]-grid_start[1]
    if x_coord > block_size * len(grid)-1 or y_coord > block_size * len(grid[0])-1 or x_coord < 0 or y_coord < 0:
        out_of_range = True 
    return (x_coord//block_size, y_coord//block_size, out_of_range)


running=True

prnt = 100
i=0
while running:
    i+=1
    SCREEN.fill(BLACK)
    mouse_pos = pygame.mouse.get_pos()
    mouse_coord = vector_to_coord(mouse_pos)
    drawGrid()
    guess = guesser.feed_forward(conv(grid_transform))["prediction"]

    drawNum(numbers[guess])

    # mouse.get_pressed --> {left_click : bool, right_click : bool, middle : bool}
    # grid transform : 255 = black
    # grid : 255 = white
    if pygame.mouse.get_pressed()[0]:
        if not mouse_coord[2]:
            grid[mouse_coord[0]][mouse_coord[1]] = 0
            grid_transform[0][mouse_coord[1]][mouse_coord[0]] = 255

    if pygame.mouse.get_pressed()[2]:
        if not mouse_coord[2]:
            grid[mouse_coord[0]][mouse_coord[1]] = 255
            grid_transform[0][mouse_coord[1]][mouse_coord[0]] = 0
            
    if pygame.mouse.get_pressed()[1]:
        grid = [[255 for x in range(len(grid))] for y in range(len(grid[0]))]
        grid_transform = np.zeros(shape = (1, grid_size[0], grid_size[1]))

        # make reseting shit laptop friendly
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN: 
            if event.key == pygame.K_SPACE:
                grid = [[255 for x in range(len(grid))] for y in range(len(grid[0]))]
                grid_transform = np.zeros(shape = (1, grid_size[0], grid_size[1]))
                
    pygame.display.flip()


