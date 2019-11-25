import matplotlib.pyplot as plt
import math
import random

#initialize theta and R
currentTheta = 0
currentR = 0
#set up polar plt
ax = plt.subplot(111, projection="polar")
ax.set_ylim([0,30])
#enable interactive mode so the plot doesn't close and open and is non-blocking
plt.ion()
plt.show()
#just for demonstration
countingUp = True
while(1):
    #***demo stuff***
    #currentTheta = random.randrange(0,359)
    currentTheta += 5 
    #currentR = random.randrange(0,10)
    if(currentR == 30):
        countingUp = False
    elif(currentR == 0):
        countingUp = True
    if(countingUp):
        currentR += 0.75
    else:
        currentR -= 0.75
    #***demo stuff end***
    #set theta and r
    theta = [0,math.radians(currentTheta)]
    r = [0,currentR]
    #clear plot and prep for redraw
    ax.cla()
    #set limit on polar plot range
    ax.set_ylim([0,30])
    #tell the user what the range is at the moment in the bottom left corner
    ax.annotate('Range: {:0.2f}'.format(currentR) + " Feet",
            xy=(currentTheta, currentR),  # theta, radius
            xytext=(0.05, 0.05),    # fraction, fraction
            textcoords='figure fraction',
    #        arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='bottom',
            )
    #plot first a red diamond and then a red line, both for position
    ax.plot(theta, r,'^r')
    ax.plot(theta, r,'-r')
    #give the plot a moment to breathe and do its processing
    plt.pause(0.05)
