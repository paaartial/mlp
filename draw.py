import pygame, sys, time

import numpy as np

from network import Network
from run import load_network, conv

# from network import Network
# from training import load_network

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
window_height = 800
window_width = 1000
block_size=20
grid_size=[10, 10]
grid_start=[int(0.12*window_width), int(0.13*window_height)]
grid = [[255 for x in range(28)] for y in range(28)]
grid_transform = np.ndarray(shape = (28, 28, 1))

mouse_down = False

guesser = load_network("train_complete", new_name="guesser")

global SCREEN, CLOCK
pygame.init()
SCREEN = pygame.display.set_mode((window_width, window_height))
CLOCK = pygame.time.Clock()
SCREEN.fill(BLACK)


def drawGrid(grid):
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            y_coord=y*block_size + grid_start[1]
            x_coord=x*block_size + grid_start[0]
            rect = pygame.Rect(x_coord, y_coord, block_size, block_size)
            pygame.draw.rect(SCREEN, (grid[x][y], grid[x][y], grid[x][y]), rect)

def vector_to_coord(pos):
    out_of_range = False
    x_coord = pos[0]-grid_start[0]
    y_coord = pos[1]-grid_start[1]
    if x_coord > block_size * len(grid)-1 or y_coord > block_size * len(grid[0])-1 or x_coord < 0 or y_coord < 0:
        out_of_range = True 
    return (x_coord//block_size, y_coord//block_size, out_of_range)
    

running=True
if_print=1000
i=0

while running:
    SCREEN.fill(BLACK)
    i+=1
    mouse_pos = pygame.mouse.get_pos()
    drawGrid(grid)
    if i % if_print == 0:
        print(grid_transform)
        print(guesser.feed_forward(conv(grid_transform))["prediction"])
    if mouse_down:
        mouse_coord = vector_to_coord(mouse_pos)
        if not mouse_coord[2]:
            try:
                grid[mouse_coord[0]][mouse_coord[1]] = 0
                grid_transform[mouse_coord[0]][mouse_coord[1]] = 255

            except IndexError:
                print(mouse_coord)
                
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_down=False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down=True


    pygame.display.update()


