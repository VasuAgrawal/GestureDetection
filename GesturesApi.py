import cv2
import numpy as np
import time
import os
import defaultGesturesLoader
from gesture import Gesture
import random
import itertools
from line import Line

class GestureProcessor(object):
    def __init__(self, gestureFile = "gestureData.txt"):
        self.cap = cv2.VideoCapture(0)
        self.cameraWidth = 1280
        self.cameraHeight = 720
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.cameraWidth)
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.cameraHeight)
        self.handCenterPositions = []
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

    # http://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-
    #increase-color-image-contrast-with-opencv-in-python-c
    @staticmethod
    def boostContrast(img):
        maxIntensity = 255.0
        phi = 1
        theta = 1
        boosted = (maxIntensity / phi) * (img/(maxIntensity/theta)) ** 2
        return np.array(boosted, np.uint8)

    @staticmethod
    def threshold(img):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        value = (31, 31)
        blurred = cv2.GaussianBlur(grey, value, 0)
        retVal, thresh = cv2.threshold(blurred, 0, 255,
                                        cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    def setContours(self, img):
        self.contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)

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
        self.handContour = self.contours[index]

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
        self.handCenterPositions += [tuple(self.centerAvg)]
        if len(self.handCenterPositions) > 10:
            self.canDoGestures = True
        else: self.canDoGestures = False

    def findHandCenter(self):
        def weightedAvg(num1, num2, weight):
            return (num1 * weight + num2) / (weight + 1)

        self.handXCenterMoment = int(self.handMoments["m10"]/
                                        self.handMoments["m00"])
        self.handYCenterMoment = int(self.handMoments["m01"]/
                                        self.handMoments["m00"])
        self.handMoment = (self.handXCenterMoment, self.handYCenterMoment)
        defectPoints = []
        minDistance = 500
        for i in self.defects:
            if i[0][3] > minDistance:
                defectPoints.append(self.handContour[i[0][2]])
        defectPoints = np.array(defectPoints, dtype = np.int32)
        combinations = itertools.combinations(defectPoints, 3) # make a triangle
        # combinations = itertools.combinations(self.hullPoints, 3)
        self.centerCandidates = []
        self.centerAvg = [0, 0]
        self.radAvg = 0
        circleNum = 0
        for combination in combinations:
            tPoints = []
            for point in combination:
                tPoints.extend(point)
            circle = self.maxInscribedCircle(np.array(tPoints))
            if circle != None:
                center, radius = circle
            else:
                continue
            center = np.array(center)
            if center[0] >= self.minX and center[0] <= self.minX + self.handWidth and center[1] >= self.minY and center[1] <= self.minY + self.handHeight and radius * 2 < min(self.handWidth, self.handHeight):
                self.centerCandidates.append(np.array([radius, center[0], center[1]], dtype=np.int32))
                self.centerAvg = [weightedAvg(self.centerAvg[0], center[0], circleNum), weightedAvg(self.centerAvg[1], center[1], circleNum)]
                self.radAvg = weightedAvg(self.radAvg, radius, circleNum)
                circleNum += 1
        if len(self.handMomentPositions) > 0:
            if Gesture.distance(self.handMoment, self.handMomentPositions[-1]) < 5.0:
                # staying roughly in place, so only chage the new one by small fraction
                self.centerAvg = [weightedAvg(self.handCenterPositions[-1][0], self.centerAvg[0], 10), weightedAvg(self.handCenterPositions[-1][1], self.centerAvg[1], 100)]


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
            return np.array(center, dtype = np.int32), int(Line.distance(center, points[0]))
        else:
            return None

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
            xPoints = [pt[0] for pt in self.handCenterPositions[val:-1]]
            yPoints = [pt[1] for pt in self.handCenterPositions[val:-1]]
            xAvg = np.average(xPoints)
            yAvg = np.average(yPoints)
            factor = 0.04
            for x, y in self.handCenterPositions[-(searchLength + 1):-1]:
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
        # print time.time() - prevTime,; prevTime = time.time()
        self.original = cv2.flip(self.original, 1)
        # print time.time() - prevTime,; prevTime = time.time()
        # self.boostContrast = GestureProcessor.boostContrast(self.original)
        # print time.time() - prevTime,; prevTime = time.time()
        self.thresholded = GestureProcessor.threshold(self.original)
        # print time.time() - prevTime,; prevTime = time.time()
        self.setContours(self.thresholded.copy())
        # print time.time() - prevTime,; prevTime = time.time()
        self.findHandContour()
        # print time.time() - prevTime,; prevTime = time.time()
        self.setHandDimensions()
        # print time.time() - prevTime,; prevTime = time.time()
        self.analyzeHandCenter()
        # print time.time() - prevTime,; prevTime = time.time()
        self.determineIfGesture()
        # print time.time() - prevTime; prevTime = time.time()

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
        cv2.circle(self.drawingCanvas, (self.handXCenterMoment, self.handYCenterMoment), 10, (255, 255, 255), -2)
        if len(self.recentPositions) != 0:
            for i in xrange(len(self.recentPositions)):
                cv2.circle(self.drawingCanvas, self.recentPositions[i], 5, (255, 25*i, 25*i), -1)

    def drawCircles(self):
        # for i in xrange(len(self.centerCandidates)-1, len(self.centerCandidates)-1-min(len(self.centerCandidates), 10), -1):
        #     cv2.circle(self.drawingCanvas, (self.centerCandidates[i][1], self.centerCandidates[i][2]), self.centerCandidates[i][0], (255, 0, 255), 3)
        #     cv2.circle(self.drawingCanvas, (self.centerCandidates[i][1], self.centerCandidates[i][2]), 2, (255, 0, 255), -1)
        cv2.circle(self.drawingCanvas, tuple(self.centerAvg), self.radAvg, (255, 0, 255), 3)
        cv2.circle(self.drawingCanvas, tuple(self.centerAvg), 10, (255, 0, 255), -1)

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
            self.drawBubbles(defectPoints, (0, 0, 255), width = 4)

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
        # cv2.imshow('Original', self.original)
        # cv2.imshow('HandContour', self.drawingCanvas)