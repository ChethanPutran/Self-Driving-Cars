from os import times
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import random
import numpy as np
import math


# ### System parameters

#No. of trials
TRIALS = 5

#No. of boxes
NO_BOXES = 3

#Acceleration due to gravity
GRAVITY = 9.8

#Mass of the box = kg
BOX_MASS = 10

#Inclined angle of the plane
INCLINED_ANGLE =  0

MAX_WINDOW_WIDTH = 200
MAX_WINDOW_HEIGHT = 110
PADDING = 5

#Time of animation(in sec)
DELTA_T = 0.05
TIME_LENGTH = 5

INIT_X = MAX_WINDOW_WIDTH - PADDING
INIT_Y = (MAX_WINDOW_WIDTH - PADDING)*np.tan(INCLINED_ANGLE)

# ### PID Constants
Kp = 300
Kd = 300
Ki = 10


class Box:
    def __init__(self,times,n_steps,n_trials,width=10,height=10):
        self.WIDTH = width
        self.HEIGHT = height
        self.times = times
        self.posX = np.ones((n_steps,n_trials),dtype=np.float16)
        self.posY = np.zeros((n_steps,n_trials),dtype=np.float16)
        self.counter = 0
        self.trial = 0
        self.n_trials = n_trials
        self.initialize()

    def initialize(self):
        self.counter = 0
        self.x = random.randrange(10,MAX_WINDOW_WIDTH-10)
        init_y = random.randrange(70,MAX_WINDOW_HEIGHT)
        self.posX[:,self.trial] *= self.x
        self.posY[:,self.trial] = init_y - GRAVITY*0.5*(self.times**2)

    def newTrail(self):
        self.trial += 1
        if (self.trial < self.n_trials):
            self.initialize()
    
    def getPositon(self):
        return self.posX[self.counter,self.trial]
        
class PID:
    def __init__(self,n_steps,n_trials,dt,kp=300,kd=300,ki=10):
        self.errors = np.zeros((n_steps,n_trials),dtype=np.float16)
        self.errors_dot = np.zeros((n_steps,n_trials),dtype=np.float16)
        self.errors_integ = np.zeros((n_steps,n_trials),dtype=np.float16)
        self.counter = 0
        self.dt = dt
        self.K_p = kp
        self.K_d = kd
        self.K_i = ki
        self.trials = n_trials
        self.trial = 0

    def __call__(self, error):
        if not self.counter:
            self.errors[self.counter,self.trial] = error
            self.counter += 1
            return self.K_p * self.errors[0,self.trial]

    
        #Updating error
        self.errors[self.counter,self.trial] = error

        #Updating de/dt
        self.errors_dot[self.counter,self.trial] = (self.errors[self.counter,self.trial] - self.errors[self.counter-1,self.trial])/self.dt

        #Updating integral(e)
        self.errors_integ[self.counter,self.trial] = (self.errors[self.counter,self.trial] + self.errors[self.counter-1,self.trial])*self.dt/2

        F = self.K_p * error + self.K_d*self.errors_dot[self.counter,self.trial] + self.K_i*self.errors_integ[self.counter,self.trial]
        self.counter += 1

        return F
    
    def newTrail(self):
        if self.trial < self.trials:
            self.counter = 0
            self.trial += 1

class System:
    def __init__(self,time_length,delta_t,trails=1,mass=100):
        #Time steps array
        self.times = np.arange(0,time_length+delta_t,delta_t)
        self.n_steps = len(self.times)
        self.trails = trails
        self.dt = delta_t
        self.positions = np.zeros((trails,self.n_steps,2),dtype=np.float16)
        self.displacements = np.zeros((self.n_steps,trails),dtype=np.float16)
        self.velocities =  np.zeros((trails,self.n_steps,2),dtype=np.float16)
        self.accelerations  = np.zeros((trails,self.n_steps,2),dtype=np.float16)
        self.mass = mass
        self.weight = -mass*GRAVITY
        self.weight_tangential = self.weight*np.sin(INCLINED_ANGLE)
        self.WIDTH = 30
        self.HEIGHT = 10
        self.box  = Box(self.times,self.n_steps,trails)
        self.pid = PID(n_steps=self.n_steps,n_trials=trails,dt=self.dt)
        self.trail = 0

    def initialize(self,init_pos=(0,0),init_velocity=0,init_acc=0):
        self.positions[self.trail][0] = init_pos 
        self.displacements[self.trail][0] = math.sqrt(init_pos[0]**2 + init_pos[1]**2)
        self.velocities[self.trail][0] = init_velocity
        self.accelerations[self.trail][0] = init_acc

    def simulate(self):
        while self.trail  < self.trails:
            got = False
            for i in range(self.n_steps):
                #Computing horizontal error
                error = self.box.getPositon() - self.positions[self.trail][i-1][0]

                #Calculating force from PID conteoller
                F_pid = self.pid(error)
                F_net = F_pid + self.weight_tangential

                self.accelerations[self.trail][i][0] = F_net / self.mass
                self.velocities[self.trail][i][0] = self.velocities[self.trail][i-1][0] + (self.accelerations[self.trail][i-1][0]+self.accelerations[self.trail][i][0])*self.dt/2 
                self.displacements[i,self.trail] = self.displacements[i-1,self.trail] + (self.velocities[self.trail][i-1][0]+self.velocities[self.trail][i][0])*self.dt/2 

                self.positions[self.trail][i] = (self.displacements[i-1,self.trail]*np.cos(INCLINED_ANGLE),
                                    self.displacements[i-1,self.trail]*np.sin(INCLINED_ANGLE))
                
                #Catching the box
                #Checking the horizontal distance
                if (got == True) or ((self.positions[self.trail][i,0] - self.box.x) < ((self.box.WIDTH + self.WIDTH)/2)):
                ##Checking vertical distance
                    if (got == True) or ((self.box.posY[i,self.trail] - self.positions[self.trail][i,1]) > 0) and  (self.box.posY[i,self.trail] < (self.positions[self.trail][i,1] + self.box.HEIGHT/2 + self.HEIGHT/2)):
                        got = True
                        self.box.posX[i,self.trail] = self.positions[self.trail][i,0]
                        self.box.posY[i,self.trail] = self.positions[self.trail][i,1]+(self.HEIGHT+self.box.HEIGHT)/2
            
            self.newTrial()

    def newTrial(self):
        self.trail += 1
        self.box.newTrail()
        self.pid.newTrail()

def simulation(system):
    
    def updatePlot(num):
        trial = int(num/TIMES_LEN)
        idx = (num - trial*TIMES_LEN)

        #Displaying Trial No
        text.set_text("Trial {} ,Time : {}s".format(trial+1,round(system.times[idx],2)))

        #Resetting system position 
        sys_obj.set_xy((system.positions[trial][idx,0]-system.WIDTH/2,system.positions[trial][idx,1]-system.HEIGHT/2))
        
        #Resetting box position
        box_obj.set_xy((system.box.posX[idx,trial]-system.box.WIDTH/2,system.box.posY[idx,trial]-system.box.HEIGHT/2))

        #Updating displacement plot
        dispPlot.set_data([system.times[0:idx]],[system.displacements[0:idx,trial]])     

        # #Updating velocity plot             
        velocityPlot.set_data([system.times[0:idx]],[system.velocities[trial][0:idx,0]])           

        # #Updating acceleration plot  
        accelerationPlot.set_data([system.times[0:idx]],[system.accelerations[trial][0:idx,0]])     

        # #Updating error plot             
        errorPlot.set_data([system.times[0:idx]],[system.pid.errors[0:idx,trial]]) 

        # #Updating error-integration plot                 
        eintegPlot.set_data([system.times[0:idx]],[system.pid.errors_integ[0:idx,trial]])  

        # #Updating error-diff plot                
        ederivPlot.set_data([system.times[0:idx]],[system.pid.errors_dot[0:idx,trial]])   

        return text,sys_obj,box_obj,platform,dispPlot,velocityPlot,accelerationPlot,errorPlot,eintegPlot,ederivPlot

    #Creating plots aand animation
    TIMES_LEN = len(system.times)
    frame_amount = TIMES_LEN*TRIALS
    figure = plt.figure(figsize=(11,6),dpi=90,facecolor=(0,0,0))

    plt.rcParams.update({
        'text.color': "white",
        'axes.labelcolor': "green",
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.spines.top':False,
        'axes.spines.bottom':True,
        'axes.spines.left':True,
        'axes.spines.right':False,
        'axes.edgecolor':'white'
        })

    gs = GridSpec(4,3)

    #Main window (3*2)
    mainAx = figure.add_subplot(gs[0:3,0:2],facecolor=(0,0,0))
    max_width = MAX_WINDOW_WIDTH + system.WIDTH
    min_width = -system.WIDTH - 5
    max_height = MAX_WINDOW_HEIGHT
    min_height = -system.HEIGHT-5

    text = plt.text(10, MAX_WINDOW_HEIGHT-10, 'Trial 1 ,Time : 0.00s', fontsize = 14, bbox = dict(facecolor = '#ccc', alpha = 0.5))
    
    plt.xlim(min_width,max_width)
    plt.ylim(min_height,max_height)
    plt.xticks(np.arange(0,MAX_WINDOW_WIDTH+1,20))
    plt.yticks(np.arange(0,MAX_WINDOW_HEIGHT+1,10))
    plt.grid(False)

    sys_obj = Rectangle((0, 0),system.WIDTH, system.HEIGHT,color ='#8B4513')
    box_obj = Rectangle((0, MAX_WINDOW_HEIGHT-system.box.HEIGHT),system.box.WIDTH, system.box.HEIGHT,color ='#DEB887')
    
    platform , = mainAx.plot([0,MAX_WINDOW_WIDTH],[0,MAX_WINDOW_WIDTH*np.tan(INCLINED_ANGLE)],'k',linewidth=4,zorder=0)
    mainAx.add_patch(sys_obj)
    mainAx.add_patch(box_obj)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = True, bottom = True)
    mainAx.spines['left'].set_visible(False)
 
    #Displacement Plot(1*1)
    dispAx = figure.add_subplot(gs[0,2],facecolor=(0,0,0))
    dispPlot,=dispAx.plot([],[],c='orange',linewidth=2)
    plt.xlim(0,TIME_LENGTH)
    plt.ylim(np.min(system.displacements)-np.abs(np.min(system.displacements))*0.2,np.max(system.displacements) + np.abs(np.max(system.displacements))*0.2)
    plt.grid(False)
    plt.xlabel('Time (in Sec)')
    plt.title('Displ. on rails [m]')

    #Velocity Plot(1*1)
    velAx = figure.add_subplot(gs[1,2],facecolor=(0,0,0))
    velocityPlot,=velAx.plot([],[],c='orange',linewidth=2)
    plt.xlim(0,TIME_LENGTH)
    plt.ylim(np.min(system.velocities)-np.abs(np.min(system.velocities))*0.2,np.max(system.velocities)+np.abs(np.max(system.velocities))*0.2)
    plt.grid(False)
    plt.xlabel('Time (in Sec)')
    plt.title('Velocity on rails [m/s]')

    #Acceleration Plot(1*1)
    acceAx=figure.add_subplot(gs[2,2],facecolor=(0,0,0))
    accelerationPlot,=acceAx.plot([],[],c='orange',linewidth=2)
    plt.xlim(0,TIME_LENGTH)
    plt.ylim(np.min(system.accelerations)-np.abs(np.min(system.accelerations))*0.2,np.max(system.accelerations) +np.abs(np.max(system.accelerations))*0.2)
    plt.grid(False)
    plt.xlabel('Time (in Sec)')
    plt.title('Accel. on rails [m/s^2] = F_net/m_platf.')

    #Error Plot (1*1)
    eAx=figure.add_subplot(gs[3,0],facecolor=(0,0,0))
    errorPlot,=eAx.plot([],[],c='orange',linewidth=2,)
    plt.xlim(0,TIME_LENGTH)
    plt.ylim(np.min(system.pid.errors)-np.abs(np.min(system.pid.errors))*0.2,np.max(system.pid.errors) + np.abs(np.max(system.pid.errors))*0.2)
    plt.grid(False)
    plt.xlabel('Time (in Sec)')
    plt.title('Horizontal error [m]')

    #Error Derivative plot(1*1)
    ederivAx=figure.add_subplot(gs[3,1],facecolor=(0,0,0))
    ederivPlot,=ederivAx.plot([],[],c='orange',linewidth=2)
    plt.xlim(0,TIME_LENGTH)
    plt.ylim(np.min(system.pid.errors_dot)-np.abs(np.min(system.pid.errors_dot))*0.2,np.max(system.pid.errors_dot)+np.abs(np.max(system.pid.errors_dot))*0.2+10)
    plt.grid(False)
    plt.xlabel('Time (in Sec)')
    plt.title('Change of horiz. error [m/s]')

    #Error integral plot(3*1)
    eintegAx=figure.add_subplot(gs[3,2],facecolor=(0,0,0))
    eintegPlot,=eintegAx.plot([],[],c='orange',linewidth=2)
    plt.xlim(0,TIME_LENGTH)
    plt.ylim(np.min(system.pid.errors_integ)-np.abs(np.min(system.pid.errors_integ))*0.2,np.max(system.pid.errors_integ)+np.abs(np.max(system.pid.errors_integ))*0.2)
    plt.grid(False)
    plt.xlabel('Time (in Sec)')
    plt.title('Sum of horiz. error [m*s]')

    figure.tight_layout(pad=1.0)

    #Animation
    animation = anim.FuncAnimation(figure,updatePlot,repeat=False,frames=frame_amount,interval=20,blit=True)
    plt.show()

#Creating system
system = System(TIME_LENGTH,DELTA_T,trails=TRIALS)

#Initializing the sysytem
system.initialize()

#Simulating the environment
system.simulate()

#Simulating results
simulation(system)
