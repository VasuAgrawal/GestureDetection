import cv2
import numpy as np
import time
import math
import sys
from os import path

class Gesture(object):
    __GestureMaxDim = 1024.0 # Nice round number
    # Keys for the return values of classify gesture
    totalError = "totalError"
    minDistance = "minDistance"
    maxDistance = "maxDistance"
    totalDistance = "totalDistance"
    distanceList = "distanceList"
    distanceRange = "distanceRange"

    def __init__(self, points, name = ""):
        self.points = np.array(points, dtype = np.float)
        self.points = Gesture.normalizePoints(self.points)
        scaleFactor = (Gesture.__GestureMaxDim /
                        Gesture.maxDim(self.points)["maxDim"])
        self.points *= scaleFactor
        self.distance = Gesture.curveLength(self.points)
        self.distance, self.distanceIndices = Gesture.curveLengthDI(self.points)
        self.name = name

    @staticmethod
    def curveLength(points):
        distance = 0
        for i in xrange(len(points)-1):
            distance += (abs(points[i][0] - points[i+1][0]) ** 2 + abs(points[i][1] - points[i+1][1]) ** 2) ** 0.5
        return distance

    @staticmethod
    def normalizePoints(points):
        return points - points[0]

    @staticmethod
    def maxDim(points):
        xMin, xMax, yMin, yMax = sys.maxsize, -sys.maxsize, sys.maxsize, -sys.maxsize
        for x, y in points:
            if x < xMin: xMin = x
            if x > xMax: xMax = x
            if y < yMin: yMin = y
            if y > yMin: yMax = y
        return {"xMin":xMin, "xMax":xMax, "yMin":yMin, "yMax":yMax,
                "maxDim": max(yMax-yMin, xMax-xMin)}

    @staticmethod
    # Takes points, returns an array of the same length with indices matching 
    # cumulative distance, to take linearization indices from.
    def curveLengthDI(points):
        indices = np.empty(len(points))
        cumulativeDistance = 0
        indices[0] = 0
        for i in xrange(1, len(points)):
            cumulativeDistance += Gesture.distance(points[i], points[i-1])
            indices[i] = cumulativeDistance
        return cumulativeDistance, indices

    @staticmethod
    def distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    @staticmethod
    def compareGestures(template, humanGesture):
        # print template.name
        # print template.points
        # print template.distanceIndices
        # print
        # print humanGesture.name
        # print humanGesture.points
        # print humanGesture.distanceIndices
        def findIndices(templateDistance):
            if templateDistance > template.distanceIndices[-1]:
                return len(template.distanceIndices) - 2, len(template.distanceIndices) - 1
            elif templateDistance < template.distanceIndices[0]:
                return 0, 1
            start = 0
            end = len(template.distanceIndices)
            while True:
                mid = (start + end) / 2
                if template.distanceIndices[mid] == templateDistance:
                    return max(mid - 1, 0), min(mid + 1,
                                                len(template.distanceIndices)-1)
                elif start == end:
                    if (templateDistance.distanceIndices[start] <
                        templateDistance):
                        return start, min(start + 1,
                                            len(template.distanceIndices)-1)
                    else:
                        return max(start - 1, 0), start
                elif abs(start - end) == 1:
                    return (min(start, end), max(start, end))
                elif template.distanceIndices[mid] < templateDistance:
                    start = mid
                else:
                    end = mid

        def linearizeTemplate(templateDistance):
            minIndex, maxIndex = findIndices(templateDistance)
            distanceDiff = (template.distanceIndices[maxIndex] -
                            template.distanceIndices[minIndex])
            templateDistance -= template.distanceIndices[minIndex]
            scale = templateDistance / distanceDiff
            change = template.points[maxIndex] - template.points[minIndex]
            change *= scale
            return template.points[minIndex] + change

        # Can probably do something with a normal distribution and whatever
        totalDistance = 0
        totalError = 0
        distances = []
        for i in xrange(len(humanGesture.distanceIndices)):
            toFind = (template.distance * 
                        humanGesture.distanceIndices[i] /
                        humanGesture.distance) 
            comparePoint = linearizeTemplate(toFind)
            distance = Gesture.distance(comparePoint, humanGesture.points[i])
            # print comparePoint, humanGesture.points[i], distance
            totalDistance += distance
            distances += [distance]
            totalError += distance ** 2 # come up with a better error function?
        minDistance = min(distances)
        maxDistance = max(distances)
        distanceRange = maxDistance - minDistance
        assessment = {Gesture.distanceList: distances,
                Gesture.minDistance: minDistance,
                Gesture.maxDistance: maxDistance,
                Gesture.totalDistance: totalDistance,
                Gesture.distanceRange: distanceRange,
                Gesture.totalError: totalError}
        # print assessment
        return assessment
        # Gesture distance determines number of partitions of the template curve
        # Subsequent distances form indices

    def action(self):
        print self.name

class HandProcessor(object):
    def __init__(self, gestureFile = "gestureData.txt"):
        self.cap = cv2.VideoCapture(0)
        self.cameraWidth = 1920
        self.cameraHeight = 1080
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
        self.initGestures()

    def initGestures(self):
        if path.isfile(self.gestureFile):
            self.loadGesturesFromFile()
        else:
            self.loadDefaultGestures()
        self.loadDefaultGestures()
        # self.gestures = self.gestures[:2]

    def loadGesturesFromFile(self):
        self.gestures = []
        with open(self.gestureFile, 'r') as fin:
            data = fin.read().split('\n')
            if len(data) == 0:
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
                        self.gestureName = ""
                        self.gesturePoints = []
                    else:
                        gesturePoints.append(map(float, item.split()))
            fin.close()


    # Initiate some default gesures in the event that no gesture file was found
    def loadDefaultGestures(self):
        self.gestures = []
        hLineLR = Gesture([(-x, 0) for x in xrange(17)],
            name="Horizontal Line Right to Left")
        self.gestures.append(hLineLR)
        hLineRL = Gesture([(x, 0) for x in xrange(17)],
            name="Horizontal Line Left to Right")
        self.gestures.append(hLineRL)
        # Y is reversed, remember?
        # let's try something more complicated, like a circle:
        circlePoints = [(10*math.cos(t), 10*math.sin(t)) \
                                    for t in np.linspace(0, 2*math.pi, num=256)]
        ccwCircle = Gesture(circlePoints, name="CW Circle")
        self.gestures.append(ccwCircle)
        circlePoints = [(10*math.cos(t), -10*math.sin(t)) \
                                    for t in np.linspace(0, 2*math.pi, num=256)]
        cwCircle = Gesture(circlePoints, name="CCW Circle")
        self.gestures.append(cwCircle)

    def close(self):
        self.cap.release()
        self.saveGestures()
        cv2.destroyAllWindows()

    def saveGestures(self):
        with open(self.gestureFile, 'w+') as fout:
            for gesture in self.gestures:
                fout.write(self.gestureHeader + gesture.name + '\n')
                for i in xrange(len(gesture.points)):
                    fout.write(str(gesture.points[i][0]) + ' ' + str(gesture.points[i][1]) + '\n')
                fout.write(self.gestureEnd + '\n')
            fout.close()

    # http://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c
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
        self.contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Currently just finds the largest contour, which seems to work to some degree
    # Should be able to replace this with a "matching" algorithm instead, from here:
    # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    def findHandContour(self):
        maxArea, index = 0, 0
        for i in xrange(len(self.contours)):
            area = cv2.contourArea(self.contours[i])
            if area > maxArea:
                maxArea = area
                index = i
        self.handContour = self.contours[index]
        self.hullHandContour = cv2.convexHull(self.handContour, returnPoints = False)
        self.defects = cv2.convexityDefects(self.handContour, self.hullHandContour)
        self.handMoments = cv2.moments(self.handContour)
        self.handXCenterMoment = int(self.handMoments["m10"]/self.handMoments["m00"])
        self.handYCenterMoment = int(self.handMoments["m01"]/self.handMoments["m00"])
        self.handCenterPositions += [(self.handXCenterMoment, self.handYCenterMoment)]
        if len(self.handCenterPositions) > 10:
            self.canDoGestures = True
        else: self.canDoGestures = False

    def analyzeHandCenter(self):
        # makes sure that there is actually sufficient data to trace over
        if len(self.handCenterPositions) > 10:
            self.recentPositions = sorted(self.handCenterPositions[-30:])
            self.x = [pos[0] for pos in self.recentPositions]
            self.y = [pos[1] for pos in self.recentPositions]
        else:
            self.recentPositions = []

    def setHandDimensions(self):
        rect = cv2.minAreaRect(self.handContour)

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
            self.gesturePoints = []

    def detemineStationary(self):
        # Figure out of the past few points have been at roughly the same position
        # If they have and there is suddenly movement, trigger the start of a gesture search
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
                if (x - xAvg) ** 2 + (y - yAvg) ** 2 > factor * min(self.cameraWidth, self.cameraHeight):
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
            # print "Calling:", self.gestures[i].name, self.humanGesture.name
            assessments[i] = Gesture.compareGestures(self.gestures[i], self.humanGesture)
            # print self.gestures[i].name
            # print assessments[i]
        errorList = [assessments[i][Gesture.totalError] for i in xrange(len(assessments))]
        index = errorList.index(min(errorList))
        # Basic elimination to figure out if result is valid
        templateGestureRatio = max((self.gestures[index].distance / self.humanGesture.distance), 
                    (self.humanGesture.distance / self.gestures[index].distance))
        distanceDiffRatio = assessments[index][Gesture.totalDistance] / min(self.gestures[index].distance, self.humanGesture.distance)
        if templateGestureRatio < 1.25 and distanceDiffRatio < 2:
            self.gestures[index].action()

        # print self.gestures[index].name, "Template Distance:", self.gestures[index].distance, "Gesture Distance:", self.humanGesture.distance, "Distance Diff:", assessments[index][Gesture.totalDistance]


    def process(self):
        while (self.cap.isOpened()):
            retVal, self.original = self.cap.read()
            self.original = cv2.flip(self.original, 1)
            self.boostContrast = HandProcessor.boostContrast(self.original)
            self.thresholded = HandProcessor.threshold(self.boostContrast)
            self.setContours(self.thresholded.copy())
            self.findHandContour()
            self.setHandDimensions()
            self.analyzeHandCenter()
            self.determineIfGesture()
            self.draw()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.close()

    def getPoint(self, index):
        if index < len(self.handContour):
            return (self.handContour[index][0][0], self.handContour[index][0][1])
        return None

# Various Drawing Methods

    def drawCenter(self):
        cv2.circle(self.drawingCanvas, (self.handXCenterMoment, self.handYCenterMoment), 10, (255, 255, 255), -2)
        if len(self.recentPositions) != 0:
            for i in xrange(len(self.recentPositions)):
                cv2.circle(self.drawingCanvas, self.recentPositions[i], 5, (255, 25*i, 25*i), -1)

    def drawHandContour(self, bubbles = False):
        cv2.drawContours(self.drawingCanvas, [self.handContour], 0, (0, 255, 0), 1)
        if bubbles:
            self.drawBubbles(self.handContour, (255, 255, 0))

    def drawHullContour(self, bubbles = False):
        hullPoints = []
        for i in self.hullHandContour:
            hullPoints.append(self.handContour[i[0]])
        hullPoints = np.array(hullPoints, dtype = np.int32)
        cv2.drawContours(self.drawingCanvas, [hullPoints], 0, (0, 0, 255), 2)
        if bubbles:
            self.drawBubbles(hullPoints, (255, 255, 255))

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
        cv2.imshow('Original', self.original)
        cv2.imshow('HandContour', self.drawingCanvas)

HandProcessor().process()

class HandProcessorSingleImage(HandProcessor):
    def __init__(self):
        self.original = cv2.imread('oneHand.jpg')

    def process(self):
        self.boostContrast = HandProcessor.boostContrast(self.original)
        self.thresholded = HandProcessor.threshold(self.boostContrast)
        self.setContours(self.thresholded.copy())
        self.findHandContour()
        self.draw()
        if cv2.waitKey(0) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()

    def draw(self):
        self.drawingCanvas = np.zeros(self.original.shape, np.uint8)
        self.drawHandContour(True)
        self.drawHullContour(True)
        self.drawDefects(True)
        cv2.imshow('HandContour', self.drawingCanvas)

# HandProcessorSingleImage().process()