from simulator import Simulator
from model import Model
import pygame

class SolenoidGraphics(object):
    def __init__(self):
        pass

    def draw(self, screen):
        x = 250
        w = 50
        h = 2

        for i in range(10):
            y = 50 - 2 * i * h # this is the bottom
            rect = (x-w/2,y-h,w,h)
            pygame.draw.rect(screen,(0,0,0),rect,0)
        
class LevitronGraphics(object):
    def __init__(self):
        self.position = [250,250]
        self.width = 30
        self.height = 20

    def draw(self, screen):
        x = self.position[0]
        y = self.position[1]
        w = self.width
        h = self.height
        rect = (x-w/2,y-h/2,w,h)
        pygame.draw.rect(screen,(0,255,0),rect,0)

    def update(self, position):
        self.position[1] = 50 + (-200 * position)
        print self.position[1]

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
    solenoid = SolenoidGraphics()
    levitron = LevitronGraphics()
    
    simulator = Simulator()
    simulator.add('solenoid', solenoid)
    simulator.add('levitron', levitron)
    simulator.run(update) ## calls callback function on every run sequence

if __name__ == "__main__":
    main()
