from gesture import Gesture
import math
import numpy as np

defaultGestures = []

def makeLines():
    pointCount = 256
    hLineLR = Gesture([(-t, 0) for t in xrange(pointCount)],
                    name="Horizontal Line Right to Left")
    hLineRL = Gesture([(t, 0) for t in xrange(pointCount)],
                    name="Horizontal Line Left to Right")
    vLineTB = Gesture([(0, t) for t in xrange(pointCount)],
                    name="Vertical Line Top to Bottom")
    vLineBT = Gesture([(0, -t) for t in xrange(pointCount)],
                    name="Vertical Line Bottom to Top")
    diagonalTLtoBR = Gesture([( t,  t) for t in xrange(pointCount)],
                            name="Diagonal Top Left to Bottom Right")
    diagonalBRtoTL = Gesture([(-t, -t) for t in xrange(pointCount)],
                            name="Diagonal Bottom Right to Top Left")
    diagonalTRtoBL = Gesture([(-t,  t) for t in xrange(pointCount)],
                            name="Diagonal Top Right to Bottom Left")
    diagonalBLtoTR = Gesture([( t, -t) for t in xrange(pointCount)],
                            name="Diagonal Bottom Left to Top Right")
    defaultGestures.append(hLineLR)
    defaultGestures.append(hLineRL)
    defaultGestures.append(vLineTB)
    defaultGestures.append(vLineBT)
    defaultGestures.append(diagonalTLtoBR)
    defaultGestures.append(diagonalBRtoTL)
    defaultGestures.append(diagonalTRtoBL)
    defaultGestures.append(diagonalBLtoTR)

def makeCircles():
    radius = 512
    pointCount = 256
    ccwCirclePoints = [(radius*math.cos(t), radius*math.sin(t)) \
                            for t in np.linspace(0, 2*math.pi, num=pointCount)]
    ccwCircle = Gesture(ccwCirclePoints, name="CW Circle")
    cwCirclePoints = [(radius*math.cos(t), -radius*math.sin(t)) \
                            for t in np.linspace(0, 2*math.pi, num=pointCount)]
    cwCircle = Gesture(cwCirclePoints, name="CCW Circle")
    defaultGestures.append(ccwCircle)
    defaultGestures.append(cwCircle)

def makeInfinity():
    scale = 30
    pointCount = 256
    lemniscatePoints = [(
        ((scale * math.sqrt(2) * math.cos(t)) / (math.sin(t) ** 2 + 1)),
        - ((scale * math.sqrt(2) * math.cos(t) * math.sin(t)) /
            (math.sin(t) ** 2 + 1)))
        for t in np.linspace(math.pi/2, 2 * math.pi + math.pi/2,
                             num=pointCount)]
    infinity = Gesture(lemniscatePoints, name="Infinity")
    defaultGestures.append(infinity)

def makeGestures():    
    makeLines()
    makeCircles()
    makeInfinity()

makeGestures()