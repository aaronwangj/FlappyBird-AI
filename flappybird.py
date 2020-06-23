"""
Python implementation of Flappy Bird
Utilizing Pygame, NEAT, collision via masks
Includes AI powered by neuroevolution artificial neural nets
NEAT: NeuroEvolution of Augmenting Topologies
Results: AI learns to play a perfect game in <5 generations
With help from Tim Ruscica
"""

import pygame
import random
import time
import os
import neat
pygame.font.init()

#Window dimensions, global variables for generation and best score
WIN_HEIGHT = 700
WIN_WIDTH = 500
GEN = 0
BEST = 0
#images and fonts
BIRD_IMG = [pygame.transform.scale2x(pygame.image.load(os.path.join("img", "b1.png"))),
pygame.transform.scale2x(pygame.image.load(os.path.join("img", "b2.png"))),
pygame.transform.scale2x(pygame.image.load(os.path.join("img", "b3.png")))]
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img", "background.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img", "ground.png")))
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("img", "pipe.png")))
STAT_FONT = pygame.font.SysFont("comicsans", 40)
#window customization
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird AI")
pygame.display.set_icon(BIRD_IMG[0])

#Bird class representing bird avatar
class Bird:
    IMG = BIRD_IMG
    ANI_TIME = 5
    ROT_VEL = 20    
    MAX_ROT = 25

    """
    Initialize bird object
    x: initial x position (int)
    y: initial y position (int)
    return: none
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.img = self.IMG[0]
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0

    """
    Controls bird movement
    return: none
    """
    def move(self):

        self.tick_count +=1

        #gravity acceleration calculation
        d = self.vel*self.tick_count+1.5*self.tick_count**2
        #terminal velocity
        if d>=16:
            d = 16
        if d<0:
            d-=2
        self.y = self.y + d
        #tilt bird upwards
        if d<0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROT:
                self.tilt = self.MAX_ROT
        #tilt bird downwards
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    """
    Controls bird's jump
    return: none
    """
    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    """
    Draws bird
    win: pygame window
    return: none
    """
    def draw(self, win):
        self.img_count += 1

        #animates bird's movement
        if self.img_count < self.ANI_TIME:
            self.img = self.IMG[0]
        elif self.img_count < self.ANI_TIME * 2:
            self.img = self.IMG[1]
        elif self.img_count < self.ANI_TIME * 3:
            self.img = self.IMG[2]
        elif self.img_count < self.ANI_TIME * 4:
            self.img = self.IMG[1]
        elif self.img_count == self.ANI_TIME * 4 + 1:
            self.img = self.IMG[0]
            self.img_count = 0

        #stops flapping if bird is falling
        if self.tilt <= -80:
            self.img = self.IMG[1]
            self.img_count = self.ANI_TIME*2
        #rotates and blits to window
        rot_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rot_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rot_image, new_rect.topleft)

    """
    Gets mask of bird
    return: mask of bird
    """
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

"""
Pipe class representing pipe obstacles
"""
class Pipe:
    GAP = 195
    VEL = 6

    """
    Initialize pipe object
    x: x position of pipe as it travels across screen
    return: none
    """
    def __init__(self,x):
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.passed = False
        self.set_height()

    """
    Sets height of pipe
    return: none
    """
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    """
    Moves pipe depending on velocity
    return: none
    """
    def move(self):
        self.x -= self.VEL

    """
    Draws pipes on window
    win: pygame window
    return: none
    """
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    """
    Checks if pipe and bird share pixel
    bird: bird object
    return: boolean on collision presence
    """
    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        t_point = bird_mask.overlap(top_mask, top_offset)
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)

        if t_point or b_point:
            return True
        return False

"""
Base class represents moving base
"""
class Base:
    IMG = BASE_IMG
    VEL = 6
    WIDTH = BASE_IMG.get_width()

    """
    Initialize base object
    y: y position
    return: none
    """
    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    """
    Moves base with pipes' velocity
    return: none
    """
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    """
    Draws base in window
    Two images, like treadmill
    return: none
    """
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

"""
Draws window with all objects present
win: pygame window
birds: all bird objects (AI)
pipes: pipe objects
score: current score
gen: current generation
best: best score
return: none
"""
def draw_window(win, birds, pipes, closest_pipe, base, score, gen, best):
    
    #background image
    win.blit(BG_IMG, (0,-125))

    for pipe in pipes:
        #draws pipes
        pipe.draw(win)

    #overlay base over pipes
    base.draw(win)

    for bird in birds:
        #draws antialiased line from bird to nearest pipe
        try:
            pygame.draw.aaline(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[closest_pipe].x + pipes[closest_pipe].PIPE_TOP.get_width()/2, pipes[closest_pipe].height))
            pygame.draw.aaline(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[closest_pipe].x + pipes[closest_pipe].PIPE_BOTTOM.get_width()/2, pipes[closest_pipe].bottom))
        except:
             pass
        # draws bird
        bird.draw(win)

    #Generation text
    gen_text = STAT_FONT.render("Generation: " + str(gen), 1, (255, 255, 255))
    win.blit(gen_text, (10, 10))

    #Number of birds alive text
    num_text = STAT_FONT.render("Birds: " + str(len(birds)), 1, (255, 255, 255))
    win.blit(num_text, (10, 45))

    #Current score text
    score_text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_text, (WIN_WIDTH - score_text.get_width() - 15, 45))

    #Best score text
    best_text = STAT_FONT.render("Best: " + str(best), 1, (255, 255, 255))
    win.blit(best_text, (WIN_WIDTH - best_text.get_width() - 15, 10))
    
    #periodically updates pygame display (animation)
    pygame.display.update()

"""
evaluates all alive bird genomes and assigns
fitness score based on distance reached
farther the bird, greater fitness, higher reproduction
return: none
"""
def eval_gen(genomes, config):
    #initial score
    score = 0
    #list of birds
    birds = []
    #list of each bird's genomes
    ge = []
    #list of neural nets assigned to each genome
    nets = []

    global GEN
    GEN += 1
    global BEST

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 0 #initial fitness of zero
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(g)

    base = Base(640)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                quit()
                break
        
        #index of closest pipe
        closest_pipe = 0
        
        #check to see which pipe to use for neural net input
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                closest_pipe = 1
        else:
            run = False
            break
        
        #reward bird for each frame it survives
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += .15
            #given bird and pipes location, returns confidence of jumping
            output = nets[x].activate((bird.y, abs(bird.y - pipes[closest_pipe].height), abs(bird.y - pipes[closest_pipe].bottom)))
            #hyperbolic tangent activation function outputs -1 to 1
            #if reasonably confident (>0.5), have bird jump
            if output[0] > 0.5:
                bird.jump()
        add_pipe = False
        rem = []
        for pipe in pipes:
            #check collision of each bird
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness-=1
                    birds.pop(x)
                    ge.pop(x)
                    nets.pop(x)
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            pipe.move()
        if add_pipe:
            #increments score
            score += 1
            #updates best score if necessary
            if score > BEST:
                BEST +=1
            #rewards bird for passing pipe
            for g in ge:
                g.fitness += 5
            #new pipe
            pipes.append(Pipe(600))
        #removes old pipe
        for r in rem:
            pipes.remove(r)
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 640 or bird.y < 0:
                birds.pop(x)
                ge.pop(x)
                nets.pop(x)
        base.move()        
        draw_window(win,birds, pipes, closest_pipe, base, score, GEN, BEST)

"""
Runs NEAT algorithm
config_path: config file path
return: none
"""
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    #creates population
    pop = neat.Population(config)
    #prints reporter in terminal to show generation stats
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    #runs given some generations (50)
    pop.run(eval_gen, 50)

#finds config file regardless of current directory
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "NEAT-config.txt")
    run(config_path)