## Import needed libraries.
import pygame
import math
from Walls import Wall
from Walls import getWalls
from Goals import Goal
from Goals import getGoals

## Declare THE POINT of reward and punishment.
GOALREWARD = 1
LIFE_REWARD = 0
PENALTY = -1

## Calculate the distance between 2 points pt1 and pt2.
def distance(pt1, pt2):
    
    return(((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5)

## This is a function that rotates a POINT around an ORIGIN by an ANGLE.
def rotate(origin,point,angle):
    
    qx = origin.x + math.cos(angle) * (point.x - origin.x) - math.sin(angle) * (point.y - origin.y)
    qy = origin.y + math.sin(angle) * (point.x - origin.x) + math.cos(angle) * (point.y - origin.y)
    q = myPoint(qx, qy)
    
    return q

## Here is a function that rotates a rectangle with vertices pt1, pt2, pt3, pt4 around a center using def rotate(origin,point,angle)
def rotateRect(pt1, pt2, pt3, pt4, angle):
    
    pt_center = myPoint((pt1.x + pt3.x)/2, (pt1.y + pt3.y)/2)
    pt1 = rotate(pt_center,pt1,angle)
    pt2 = rotate(pt_center,pt2,angle)
    pt3 = rotate(pt_center,pt3,angle)
    pt4 = rotate(pt_center,pt4,angle)
    
    return pt1, pt2, pt3, pt4

## This is a simple class presenting a point in 2D space. It has 2 properties 'x' and 'y' to save coordination
class myPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

## Class to represent a line segment in 2D space. It has two properties pt1 and pt2 to store the two endpoints of the line segment       
class myLine:
    def __init__(self, pt1, pt2):
        self.pt1 = myPoint(pt1.x, pt1.y)
        self.pt2 = myPoint(pt2.x, pt2.y)

## Class representing a ray in 2D space including 3 properties x, y (coordinates of the ray's starting point) and angle (the angle of rotation of the ray).
class Ray:
    def __init__(self,x,y,angle):
        self.x = x
        self.y = y
        self.angle = angle
    """This class provides a cast(wall) method that checks if the ray intersects the wall segment 
            and returns the intersection point ( OBJECT 'myPoint') of the ray and the wall segment if so."""
    def cast(self, wall): 
        x1 = wall.x1 
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        vec = rotate(myPoint(0,0), myPoint(0,-1000), self.angle)
        
        x3 = self.x
        y3 = self.y
        x4 = self.x + vec.x
        y4 = self.y + vec.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
        if(den == 0):
            den = 0
        else:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if t > 0 and t < 1 and u < 1 and u > 0:
                pt = myPoint(math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1)))
                return(pt)
            
class Car:
    """ Initialize a new object car and set up its first properties such as coordination,shape,image,angle,velocity """
    def __init__(self, x, y): 
        self.pt = myPoint(x, y) ## set up the  coordination and shape of the car 
        self.x = x
        self.y = y
        self.width = 14
        self.height = 30

        self.points = 0

        self.original_image = pygame.image.load("car.png").convert()
        self.image = self.original_image  ## This will reference the rotated image.
        self.image.set_colorkey((0,0,0))
        self.rect = self.image.get_rect().move(self.x, self.y)

        self.angle = math.radians(180) ## set up the angle of the car
        self.soll_angle = self.angle

        self.dvel = 1
        self.vel = 0
        self.velX = 0
        self.velY = 0
        self.maxvel = 10 # set up the maximum velocity of the car 

        self.angle = math.radians(180)
        self.soll_angle = self.angle
        ## calculate 4 vertices of the car (rectangular shape)
        self.pt1 = myPoint(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = myPoint(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = myPoint(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = myPoint(self.pt.x - self.width / 2, self.pt.y + self.height / 2)

        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

        self.distances = []
    
    """ Possible actions of the car """
    def action(self, choice):
        if choice == 0:                 # do nothing
            pass
        elif choice == 1:               # accelerate the car 
            self.accelerate(self.dvel)
        elif choice == 8:
            self.accelerate(self.dvel)  # accelerate the car and rotate clockwise 15 degree
            self.turn(1)
        elif choice == 7:
            self.accelerate(self.dvel)  # accelerate the car and rotate counterclockwise 15 degree
            self.turn(-1)
        elif choice == 4:
            self.accelerate(-self.dvel) # decelerate the car 
        elif choice == 5:
            self.accelerate(-self.dvel) # decelerate the car and rotate clockwise 15 degree
            self.turn(1)
        elif choice == 6:
            self.accelerate(-self.dvel) # decelerate the car and rotate counterclockwise 15 degree
            self.turn(-1)
        elif choice == 3:               # rotate clockwise 15 degree
            self.turn(1)
        elif choice == 2:               # rotate counterclockwise 15 degree
            self.turn(-1)
        pass
    
    """ Acceleration and deceleration function """
    def accelerate(self,dvel):
        dvel = dvel * 2

        self.vel = self.vel + dvel

        if self.vel > self.maxvel:
            self.vel = self.maxvel
        
        if self.vel < -self.maxvel:
            self.vel = -self.maxvel
        
    """ Rotation function """
    def turn(self, dir):
        self.soll_angle = self.soll_angle + dir * math.radians(15)
    
    """ This method updates the vehicle's position and rotation based on the desired velocity and rotation. 
        It also updates the image of the car according to the new angle."""
    def update(self):

        # drifting code 
        """This code handles reversing the rotation of the vehicle when the target angle ('self.soll_angle') 
            is different from the current angle (self.angle)."""
            
        if(self.soll_angle > self.angle):
            if(self.soll_angle > self.angle + math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)):
                self.angle = self.angle + math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)
            else:
                self.angle = self.soll_angle
        
        if(self.soll_angle < self.angle):
            if(self.soll_angle < self.angle - math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)):
                self.angle = self.angle - math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)
            else:
                self.angle = self.soll_angle
        
        self.angle = self.soll_angle ## Update current angle by desired angle

        vec_temp = rotate(myPoint(0,0), myPoint(0,self.vel), self.angle) ## Calculate X-axis and Y-axix velocity 
        self.velX, self.velY = vec_temp.x, vec_temp.y

        ## Update the coordinate of the car
        self.x = self.x + self.velX 
        self.y = self.y + self.velY

        self.rect.center = self.x, self.y ## the center point 

        ## Update 4 point vertices
        self.pt1 = myPoint(self.pt1.x + self.velX, self.pt1.y + self.velY)
        self.pt2 = myPoint(self.pt2.x + self.velX, self.pt2.y + self.velY)
        self.pt3 = myPoint(self.pt3.x + self.velX, self.pt3.y + self.velY)
        self.pt4 = myPoint(self.pt4.x + self.velX, self.pt4.y + self.velY)

        self.p1 ,self.p2 ,self.p3 ,self.p4  = rotateRect(self.pt1, self.pt2, self.pt3, self.pt4, self.soll_angle)

        self.image = pygame.transform.rotate(self.original_image, 90 - self.soll_angle * 180 / math.pi) ## Rotate the image of the car 
        x, y = self.rect.center  # Save its current center.
        self.rect = self.image.get_rect()  # Replace old rectangle with new rectangle.
        self.rect.center = (x, y)

    """This method performs raying(virtual sensors) from the vehicle to identify obstacles in the environment. 
        It returns observations of the distance to the nearest obstructions."""
    def cast(self, walls):

        ray1 = Ray(self.x, self.y, self.soll_angle)
        ray2 = Ray(self.x, self.y, self.soll_angle - math.radians(30))
        ray3 = Ray(self.x, self.y, self.soll_angle + math.radians(30))
        ray4 = Ray(self.x, self.y, self.soll_angle + math.radians(45))
        ray5 = Ray(self.x, self.y, self.soll_angle - math.radians(45))
        ray6 = Ray(self.x, self.y, self.soll_angle + math.radians(90))
        ray7 = Ray(self.x, self.y, self.soll_angle - math.radians(90))
        ray8 = Ray(self.x, self.y, self.soll_angle + math.radians(180))

        ray9 = Ray(self.x, self.y, self.soll_angle + math.radians(10))
        ray10 = Ray(self.x, self.y, self.soll_angle - math.radians(10))
        ray11 = Ray(self.x, self.y, self.soll_angle + math.radians(135))
        ray12 = Ray(self.x, self.y, self.soll_angle - math.radians(135))
        ray13 = Ray(self.x, self.y, self.soll_angle + math.radians(20))
        ray14 = Ray(self.x, self.y, self.soll_angle - math.radians(20))

        ray15 = Ray(self.p1.x,self.p1.y, self.soll_angle + math.radians(90))
        ray16 = Ray(self.p2.x,self.p2.y, self.soll_angle - math.radians(90))

        ray17 = Ray(self.p1.x,self.p1.y, self.soll_angle + math.radians(0))
        ray18 = Ray(self.p2.x,self.p2.y, self.soll_angle - math.radians(0))

        self.rays = []
        self.rays.append(ray1)
        self.rays.append(ray2)
        self.rays.append(ray3)
        self.rays.append(ray4)
        self.rays.append(ray5)
        self.rays.append(ray6)
        self.rays.append(ray7)
        self.rays.append(ray8)

        self.rays.append(ray9)
        self.rays.append(ray10)
        self.rays.append(ray11)
        self.rays.append(ray12)
        self.rays.append(ray13)
        self.rays.append(ray14)

        self.rays.append(ray15)
        self.rays.append(ray16)

        self.rays.append(ray17)
        self.rays.append(ray18)


        observations = [] ## contains observed values, including the distance to the obstacle and the velocity ratio.
        self.closestRays = []

        for ray in self.rays:
            closest = None #myPoint(0,0)
            record = math.inf
            for wall in walls:
                pt = ray.cast(wall)
                if pt: ## if pt (intersection exists),the distance will be calculated
                    dist = distance(myPoint(self.x, self.y),pt)
                    if dist < record: 
                        record = dist
                        closest = pt

            if closest: 
                #append distance for current ray 
                self.closestRays.append(closest)
                observations.append(record)
               
            else:
                observations.append(1000) ## distance is 1000 to describe infinite distance

        for i in range(len(observations)):
            #invert observation values 0 is far away 1 is close
            observations[i] = ((1000 - observations[i]) / 1000) ## the value is converted to range from 0 (very far) to 1 (very close).

        observations.append(self.vel / self.maxvel) ## The rate velocity is equal the current velocity/ maximum velocity
        return observations

    """The ('collision') method helps to determine if the vehicle has collided with an obstacle based on 
        checking the intersection of the lines of the vehicle and the obstacle."""
    def collision(self, wall):
        ## create 4 lines for 4 vertices of the car
        line1 = myLine(self.p1, self.p2)
        line2 = myLine(self.p2, self.p3)
        line3 = myLine(self.p3, self.p4)
        line4 = myLine(self.p4, self.p1)
        ## extract the begging and ending point of wall
        x1 = wall.x1 
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        lines = []
        lines.append(line1)
        lines.append(line2)
        lines.append(line3)
        lines.append(line4)
        ## check collision
        for li in lines:
            
            x3 = li.pt1.x
            y3 = li.pt1.y
            x4 = li.pt2.x
            y4 = li.pt2.y

            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            if(den == 0): ## there is no collision
                den = 0
            else:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

                if t > 0 and t < 1 and u < 1 and u > 0:
                    return(True) ## collision is found
        
        return(False)
    """Score(self, goal) method: This method calculates the vehicle's score based on whether the vehicle passes a goal. 
            If the vehicle approaches the target close enough, points will be added"""
    def score(self, goal):
        ## create a line from 2 vertices of the car
        line1 = myLine(self.p1, self.p3)

        vec = rotate(myPoint(0,0), myPoint(0,-50), self.angle)
        line1 = myLine(myPoint(self.x,self.y),myPoint(self.x + vec.x, self.y + vec.y))
        ## extract 2 points of the goal
        x1 = goal.x1 
        y1 = goal.y1
        x2 = goal.x2
        y2 = goal.y2
        ## extract 2 points in the car's line
        x3 = line1.pt1.x
        y3 = line1.pt1.y
        x4 = line1.pt2.x
        y4 = line1.pt2.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if(den == 0):  ## car does not touch goal
            den = 0
        else:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if t > 0 and t < 1 and u < 1 and u > 0:
                pt = math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1)) ## determine the intersection point

                d = distance(myPoint(self.x, self.y), myPoint(pt[0], pt[1])) ## calculate the distance between car and goal
                if d < 20: ## it means car went through goal
                    self.points += GOALREWARD
                    return(True)

        return(False)

    """This method resets the vehicle properties to their original state. It is used when wanting to start a new round."""
    def reset(self):

        self.x = 50
        self.y = 300
        self.velX = 0
        self.velY = 0
        self.vel = 0
        self.angle = math.radians(180)
        self.soll_angle = self.angle
        self.points = 0

        self.pt1 = myPoint(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = myPoint(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = myPoint(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = myPoint(self.pt.x - self.width / 2, self.pt.y + self.height / 2)

        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

    def draw(self, win):
        win.blit(self.image, self.rect)
  

class RacingEnv:
    """ Initialize the variables and properties of the gaming environment """
    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)

        self.fps = 120
        self.width = 1000
        self.height = 600
        self.history = []

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RACING DQN")
        self.screen.fill((0,0,0))
        self.back_image = pygame.image.load("track.png").convert()
        self.back_rect = self.back_image.get_rect().move(0, 0)
        self.action_space = None
        self.observation_space = None
        self.game_reward = 0
        self.score = 0
 
        self.reset()

    """ Resetting the racing environment. It resets the screen and regenerates the Car object, 
        the list of Wall and Goal objects, and sets the game reward to 0."""
    def reset(self):
        self.screen.fill((0, 0, 0))

        self.car = Car(50, 300)
        self.walls = getWalls()
        self.goals = getGoals()
        self.game_reward = 0

    """ The method that executes a step in a racing environment is based on the action given. 
        This method updates the state of the ('Car') object """
    def step(self, action):

        done = False
        self.car.action(action)
        self.car.update()
        reward = LIFE_REWARD

        ## Check if car passes Goal and calculate the reward point 
        index = 1
        for goal in self.goals:
            
            if index > len(self.goals):
                index = 1
            if goal.isactiv:
                if self.car.score(goal):
                    goal.isactiv = False
                    self.goals[index-2].isactiv = True
                    reward += GOALREWARD

            index = index + 1

        ## check if car crashed in the wall the game will be reset 
        for wall in self.walls:
            if self.car.collision(wall):
                reward += PENALTY
                done = True

        new_state = self.car.cast(self.walls)
        ## normalize states 
        if done:
            new_state = None

        return new_state, reward, done

    """ Represent the current state of the gaming environment on the screen by using class Car,Wall,Goal,Ray """
    def render(self, action):

        DRAW_WALLS = False
        DRAW_GOALS = False
        DRAW_RAYS = False

        pygame.time.delay(10)

        self.clock = pygame.time.Clock()
        self.screen.fill((0, 0, 0))

        self.screen.blit(self.back_image, self.back_rect)

        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        
        if DRAW_GOALS:
            for goal in self.goals:
                goal.draw(self.screen)
                if goal.isactiv:
                    goal.draw(self.screen)
        
        self.car.draw(self.screen)

        if DRAW_RAYS:
            i = 0
            for pt in self.car.closestRays:
                pygame.draw.circle(self.screen, (0,0,255), (pt.x, pt.y), 5)
                i += 1
                if i < 15:
                    pygame.draw.line(self.screen, (0,255,255), (self.car.x, self.car.y), (pt.x, pt.y), 1)
                elif i >=15 and i < 17:
                    pygame.draw.line(self.screen, (0,255,255), ((self.car.p1.x + self.car.p2.x)/2, (self.car.p1.y + self.car.p2.y)/2), (pt.x, pt.y), 1)
                elif i == 17:
                    pygame.draw.line(self.screen, (0,255,255), (self.car.p1.x , self.car.p1.y ), (pt.x, pt.y), 1)
                else:
                    pygame.draw.line(self.screen, (0,255,255), (self.car.p2.x, self.car.p2.y), (pt.x, pt.y), 1)

        ## Render controll
        pygame.draw.rect(self.screen,(255,255,255),(800, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(850, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(900, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(850, 50, 40, 40),2)
        ## Render the action of the car
        if action == 4:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40)) 
        elif action == 6:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 5:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 1:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40)) 
        elif action == 8:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 7:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 2:
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 3:
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))

        # Present the reward score
        text_surface = self.font.render(f'Points {self.car.points}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(0, 0))
        # Present the speed
        text_surface = self.font.render(f'Speed {self.car.vel*-1}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(800, 0))

        self.clock.tick(self.fps)
        pygame.display.update()

    def close(self):
        pygame.quit()