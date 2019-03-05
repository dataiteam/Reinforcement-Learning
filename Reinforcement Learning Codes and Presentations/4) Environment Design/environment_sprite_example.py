"""
Sprite
"""
# pygame template
import pygame
import random

# window size
WIDTH = 360
HEIGHT = 360
FPS = 30 # how fast game is

# colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255, 0, 0) # RGB
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Player(pygame.sprite.Sprite):
    # sprite for the player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.center =(WIDTH/2,HEIGHT/2)
        self.y_speed = 5
        self.x_speed = 3
        
    def update(self):
        self.rect.y += self.y_speed
        if self.rect.bottom > HEIGHT - 200:
            self.y_speed = -5
        if self.rect.top < 0:
            self.y_speed = 5
        
# initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("RL Game")
clock = pygame.time.Clock()

# sprite
all_sprite = pygame.sprite.Group()
player = Player()
all_sprite.add(player)

# game loop
running = True
while running:
    # keep loop running at the right speed
    clock.tick(FPS) 
    
    # process input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # update
    all_sprite.update()
    
    # draw / render(show)
    screen.fill(GREEN)
    all_sprite.draw(screen)
    # after drawing flip display
    pygame.display.flip()
    
pygame.quit()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    