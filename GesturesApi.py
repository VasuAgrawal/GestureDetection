import cv2
import numpy as np
import time
import os
import defaultGesturesLoader
from gesture import Gesture
import random
import itertools
from line import Line
from scipy.spatial.distance import pdist, squareform

class GestureProcessor(object):
    def __init__(self, gestureFile = "gestureData.txt"):
        self.cap = cv2.VideoCapture(0)
        self.cameraWidth = 1280
        self.cameraHeight = 720
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.cameraWidth)
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.cameraHeight)
        self.stationary = False
        self.record = False
        self.endGesture = False
        self.gesturePoints = []
        self.gestureFile = gestureFile
        self.gestureHeader = "Gesture Name: "
        self.gestureEnd = "END GESTURE"
        self.saveNextGesture = False
        self.lastAction = ""
        self.handMomentPositions = []
        self.handCenterPositions = []
        self.initGestures()

    def initGestures(self):
        if os.path.isfile(self.gestureFile):
            self.loadGesturesFromFile()
        else:
            self.loadDefaultGestures()

    def loadGesturesFromFile(self):
        self.gestures = []
        read = ""
        with open(self.gestureFile, 'r') as fin:
            read = fin.read()
            fin.close()
        data = read.split('\n')
        # Basic check, should replace later with bytestream instead
        if len(data) < len(self.gestureHeader):
            self.loadDefaultGestures()
        else:
            gestureName = ""
            gesturePoints = []
            cutoff = len(self.gestureHeader)
            for item in data:
                if item[:cutoff] == self.gestureHeader:
                    gestureName = item[cutoff:]
                elif item == self.gestureEnd:
                    self.gestures.append(Gesture(gesturePoints, gestureName))
                    gestureName = ""
                    gesturePoints = []
                else:
                    gesturePoints.append(map(float, item.split()))

    # Initiate some default gesures in the event that no gesture file was found
    def loadDefaultGestures(self):
        self.gestures = defaultGesturesLoader.defaultGestures

    def close(self):
        self.cap.release()
        self.saveGestures()
        cv2.destroyAllWindows()

    def saveGestures(self):
        with open(self.gestureFile, 'w+') as fout:
            for gesture in self.gestures:
                fout.write(self.gestureHeader + gesture.name + '\n')
                for i in xrange(len(gesture.points)):
                    fout.write(str(gesture.points[i][0]) + ' ' +
                                str(gesture.points[i][1]) + '\n')
                fout.write(self.gestureEnd + '\n')
            fout.close()

    @staticmethod
    def threshold(img):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        value = (31, 31)
        blurred = cv2.GaussianBlur(grey, value, 0)
        retVal, thresh = cv2.threshold(blurred, 0, 255,
                                        cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    def setContours(self, img):
        self.contours, _ = cv2.findContours(img, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)

    def getDistance(self):
        self.handDistance = (self.cameraWidth + self.cameraHeight) / float(self.palmRadius)

    # Credit for this algorithm goes to the paper which can be found at the
    # description in this link: https://www.youtube.com/watch?v=xML2S6bvMwI
    def centerByReduction(self):
        scaleFactor = 0.3
        shrunk = np.array(self.handContour * scaleFactor, dtype = np.int32)
        tx, ty, w, h = cv2.boundingRect(shrunk)
        maxPoint = None
        maxRadius = 0
        for x in xrange(w):
            for y in xrange(h):
                rad = cv2.pointPolygonTest(shrunk, (tx + x, ty + y), True)
                if rad > maxRadius:
                    maxPoint = (tx + x, ty + y)
                    maxRadius = rad
        upscaledCenter = np.array(np.array(maxPoint) / scaleFactor, dtype = np.int32)
        error = int((1 / scaleFactor) * 1.5)
        maxPoint = None
        maxRadius = 0
        for x in xrange(upscaledCenter[0] - error, upscaledCenter[0] + error):
            for y in xrange(upscaledCenter[1] - error, upscaledCenter[1] + error):
                rad = cv2.pointPolygonTest(self.handContour, (x, y), True)
                if rad > maxRadius:
                    maxPoint = (x, y)
                    maxRadius = rad
        return np.array(maxPoint)

    # Currently just finds the largest contour,
    # Should be able to replace this with a "matching" algorithm from here:
    # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/
    #py_contours_more_functions/py_contours_more_functions.html
    def findHandContour(self):
        maxArea, index = 0, 0
        for i in xrange(len(self.contours)):
            area = cv2.contourArea(self.contours[i])
            if area > maxArea:
                maxArea = area
                index = i
        self.realHandContour = self.contours[index]
        # reduce hand contour to manageable number of points
        # Thanks to http://opencvpython.blogspot.com/2012/06/contours-2-brotherhood.html
        self.realHandLen = cv2.arcLength(self.realHandContour, True)
        self.handContour = cv2.approxPolyDP(self.realHandContour, 0.001 * self.realHandLen, True)
        self.hullHandContour = cv2.convexHull(self.handContour,
                                                returnPoints = False)
        self.hullPoints = [self.handContour[i[0]] for i in self.hullHandContour]
        self.hullPoints = np.array(self.hullPoints, dtype = np.int32)
        self.defects = cv2.convexityDefects(self.handContour,
                                            self.hullHandContour)
        self.handMoments = cv2.moments(self.handContour)
        self.setHandDimensions()
        self.findHandCenter()
        self.handMomentPositions += [self.handMoment]
        self.reductionCenter = self.centerByReduction()
        self.handCenterPositions += [tuple(self.reductionCenter)]
        if len(self.handCenterPositions) > 10:
            self.canDoGestures = True
        else: self.canDoGestures = False
        self.getDistance()

    def findHandCenter(self):
        def weightedAvg(num1, num2, weight):
            return (num1 * weight + num2) / (weight + 1)

        self.handXCenterMoment = int(self.handMoments["m10"]/
                                        self.handMoments["m00"])
        self.handYCenterMoment = int(self.handMoments["m01"]/
                                        self.handMoments["m00"])
        self.handMoment = (self.handXCenterMoment, self.handYCenterMoment)
        self.palmCenter = self.centerByReduction()
        self.palmRadius = cv2.pointPolygonTest(self.handContour, tuple(self.palmCenter), True)

    def trimContour(self):
        maxRadius = min(self.handWidth, self.handHeight) / 1.25
        self.trimmed = []
        for point in self.handContour:
            if Gesture.distance(point[0], self.reductionCenter) < maxRadius:
                self.trimmed.append(point)

    def analyzeHandCenter(self):
        # makes sure that there is actually sufficient data to trace over
        if len(self.handCenterPositions) > 10:
            self.recentPositions = sorted(self.handCenterPositions[-30:])
            self.x = [pos[0] for pos in self.recentPositions]
            self.y = [pos[1] for pos in self.recentPositions]
        else:
            self.recentPositions = []

    @staticmethod
    def almostEqual(num1, num2, epsilon = 1e-10):
        return abs(num1 - num2) < epsilon

    def maxInscribedCircle(self, points):
        if not Line.colinear(points):
            AB = Line(points[0], points[1])
            AC = Line(points[0], points[2])
            a = np.array([[AB.pslope, -1], [AC.pslope, -1]])
            b = np.array([AB.pslope * AB.midpoint[0] - AB.midpoint[1], AC.pslope * AC.midpoint[0] - AC.midpoint[1]])
            center = np.linalg.solve(a, b)
            if AB.pslope == Line.inf:
                center[0] = AB.midpoint[0]
                center[1] = AC.pslope * center[0] - AC.pslope * AC.midpoint[0] + AC.midpoint[1]
            elif AC.pslope == Line.inf:
                center[0] = AC.midpoint[0]
                center[1] = AB.pslope * center[0] - AB.pslope * AB.midpoint[0] + AB.midpoint[1]
            return int(Line.distance(center, points[0])), np.array(center, dtype = np.int32)
        else:
            return None, None

    def setHandDimensions(self):
        self.minX, self.minY, self.handWidth, self.handHeight = cv2.boundingRect(self.handContour)

    def determineIfGesture(self):
        self.prevRecordState = self.record
        self.detemineStationary()
        if self.record:
            self.gesturePoints += [self.handCenterPositions[-1]]
        elif self.prevRecordState == True and not self.record:
            minGesturePoints = 5 # Should last a few frames at least
            if len(self.gesturePoints) > minGesturePoints:
                gestureIndex = self.classifyGesture()
                if gestureIndex != None:
                    self.gestures[gestureIndex].action()
                    self.lastAction = self.gestures[gestureIndex].name
                elif gestureIndex == None and self.saveNextGesture:
                    self.addRecordedGesture()
                    self.saveNextGesture = False
            self.gesturePoints = []

    def saveNext(self):
        self.saveNextGesture = True

    def addRecordedGesture(self):
        gestureName = ""
        while True:
            gestureName = "".join([chr(random.randint(ord('a'), ord('z'))) \
                                    for i in xrange(20)])
            if gestureName not in self.getGestureNames():
                break
        newGesture = Gesture(self.gesturePoints, name=gestureName)
        self.gestures.append(newGesture)
        print "RECORDED NEW ONE", gestureName
        self.lastAction = gestureName
        return gestureName

    def detemineStationary(self):
        # Figure out of the past few points have been at roughly same position
        # If they have and there is suddenly movement,
        # trigger the start of a gesture search
        searchLength = 3 # 3 frames should be enough
        val = -1 * (searchLength + 1)
        if self.canDoGestures:
            xPoints = [pt[0] for pt in self.handMomentPositions[val:-1]]
            yPoints = [pt[1] for pt in self.handMomentPositions[val:-1]]
            xAvg = np.average(xPoints)
            yAvg = np.average(yPoints)
            factor = 0.04
            for x, y in self.handMomentPositions[-(searchLength + 1):-1]:
                # if any point is further further from the average:
                if (x-xAvg)**2 + (y-yAvg)**2 > factor * min(self.cameraWidth,
                                                            self.cameraHeight):
                    # If previous not moving, start recording
                    if self.stationary:
                        self.record = True
                    self.stationary = False
                    self.stationaryTimeStart = time.time()
                    return
            # Not previously stationary but stationary now
            if not self.stationary:
                self.record = False
            self.stationary = True

    def classifyGesture(self):
        minError = 2**31 - 1 # a large value
        minErrorIndex = -1
        self.humanGesture = Gesture(self.gesturePoints, "Human Gesture")
        likelihoodScores = [0] * len(self.gestures)
        assessments = [{}] * len(self.gestures)
        for i in xrange(len(self.gestures)):
            assessments[i] = Gesture.compareGestures(self.gestures[i],
                                                        self.humanGesture)
        errorList = [assessments[i][Gesture.totalError] \
                        for i in xrange(len(assessments))]
        index = errorList.index(min(errorList))
        # Basic elimination to figure out if result is valid
        templateGestureRatio = max((self.gestures[index].distance /\
                                    self.humanGesture.distance), 
                                    (self.humanGesture.distance /\
                                        self.gestures[index].distance))
        distanceDiffRatio = assessments[index][Gesture.totalDistance] /\
                                min(self.gestures[index].distance,
                                    self.humanGesture.distance)
        if templateGestureRatio < 1.25 and distanceDiffRatio < 2:
            return index

    def bind(self, gestureIndex, fn):
        self.gestures[gestureIndex].action = fn

    def getGestureNames(self):
        return [gesture.name for gesture in self.gestures]

    # importantly, changed so that it works on a tick instead
    def process(self):
        retVal, self.original = self.cap.read()
        self.original = cv2.flip(self.original, 1)
        self.thresholded = GestureProcessor.threshold(self.original)
        self.setContours(self.thresholded.copy())
        self.findHandContour()
        self.setHandDimensions()
        self.analyzeHandCenter()
        self.determineIfGesture()

    def getPoint(self, index):
        if index < len(self.handContour):
            return (self.handContour[index][0][0],self.handContour[index][0][1])
        return None

    def getRGBAThresh(self, widthScale=1, heightScale=1):
        if widthScale != 1 or heightScale != 1:
            resized = cv2.resize(self.thresholded, (0, 0), fx=widthScale,
                                    fy=heightScale)
            return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGBA)    
        return cv2.cvtColor(self.thresholded, cv2.COLOR_GRAY2RGBA)

    def getRGBAOriginal(self, widthScale=1, heightScale=1):
        if widthScale != 1 or heightScale != 1:
            resized = cv2.resize(self.thresholded, (480, 320))
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        return cv2.cvtColor(self.original, cv2.COLOR_BGR2RGBA)

    def getRGBACanvas(self, widthScale=1, heightScale=1):
        if widthScale != 1 or heightScale != 1:
            resized = cv2.resize(self.thresholded, (0, 0), fx=widthScale,
                                    fy=heightScale)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        return cv2.cvtColor(self.drawingCanvas, cv2.COLOR_BGR2RGBA)

# Various Drawing Methods

    def drawCenter(self):
        cv2.circle(self.drawingCanvas, tuple(self.reductionCenter), 10, (255, 0, 0), -2)
        if len(self.recentPositions) != 0:
            for i in xrange(len(self.recentPositions)):
                cv2.circle(self.drawingCanvas, self.recentPositions[i], 5, (255, 25*i, 25*i), -1)

    def drawCircles(self):
        cv2.circle(self.drawingCanvas, tuple(self.palmCenter), int(self.palmRadius), (0, 255, 0), 10)

    def drawHandContour(self, bubbles = False):
        cv2.drawContours(self.drawingCanvas, [self.handContour], 0, (0, 255, 0), 1)
        if bubbles:
            self.drawBubbles(self.handContour, (255, 255, 0))

    def drawHullContour(self, bubbles = False):
        cv2.drawContours(self.drawingCanvas, [self.hullPoints], 0, (0, 0, 255), 2)
        if bubbles:
            self.drawBubbles(self.hullPoints, (255, 255, 255))

    def drawDefects(self, bubbles = False):
        defectPoints = []
        minDistance = 1000
        for i in self.defects:
            if i[0][3] > minDistance:
                defectPoints.append(self.handContour[i[0][2]])
        defectPoints = np.array(defectPoints, dtype = np.int32)
        if bubbles:
            self.drawBubbles(defectPoints, (0, 0, 255), width = 10)

    def drawBubbles(self, pointsList, color = (255, 255, 255), width = 2):
        for i in xrange(len(pointsList)):
            for j in xrange(len(pointsList[i])):
                cv2.circle(self.drawingCanvas, (pointsList[i][j][0], pointsList[i][j][1]), width, color)

    def draw(self):
        self.drawingCanvas = np.zeros(self.original.shape, np.uint8)
        self.drawHandContour(True)
        self.drawHullContour(True)
        self.drawDefects(True)
        self.drawCenter()
        self.drawCircles()