from pyfirmata import Arduino, util
import time
board = Arduino("COM3")
stepsPerRev = 200
desiredSteps = 50
#pin numbers to write
s1 = 1
s2 = 2
s3 = 3
s4 = 4
board.digital[s2].write(0)
board.digital[s3].write(0)
#take 50 steps
for x in range(desiredSteps)):
  board.digital[s1].write(1)
  board.digital[s4].write(1)
  time.sleep(0.001)
  board.digital[s1].write(0)
  board.digital[s4].write(0)

