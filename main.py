from simulator import Simulator
from model import Model
import pygame

scale = 2000.

WIDTH = 900
HEIGHT = 900

class SolenoidGraphics(object):
    def __init__(self, base):
        self.r = scale * base.radius
        self.l = scale * base.length
        self.w = self.r * 2
        self.N = base.loops

    def draw(self, screen):
        x = WIDTH / 2 # middle
        w = self.w
        h = self.l / (2 * self.N)
        print h
        skip = 8
        for i in range(int(self.N)):
            if i % skip == 0:
                y = HEIGHT/2 - i * h # this is the bottom
                rect = (x-w/2,y-h,w,max(h,2))
                pygame.draw.rect(screen,(0,0,0),rect,0)
        
class LevitronGraphics(object):
    def __init__(self, base):
        self.position = [WIDTH/2,HEIGHT/2]
        self.r = scale * base.geom.r
        self.h = scale * base.geom.h

    def draw(self, screen):
        x = self.position[0]
        y = self.position[1]
        w = 2 * self.r
        h = self.h 

        rect = (x-w/2,y-h/2,w,h)
        pygame.draw.rect(screen,(0,255,0),rect,0)

    def update(self, position):
        self.position[1] = HEIGHT/2 + (-scale * position) # flip y axis coord.

model = None
solenoid = None
levitron = None

def update():
    model.update(0.005) # dt in ec
    #solenoid.update() # --> unnecessary
    levitron.update(model.position[2])

def main():
    global model, solenoid, levitron

    model = Model()
    solenoid = SolenoidGraphics(model.solenoid)
    levitron = LevitronGraphics(model.levitron)
    
    simulator = Simulator(WIDTH,HEIGHT)
    simulator.add('solenoid', solenoid)
    simulator.add('levitron', levitron)
    simulator.run(update) ## calls callback function on every run sequence

if __name__ == "__main__":
    main()
