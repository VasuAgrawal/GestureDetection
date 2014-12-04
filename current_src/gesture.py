import numpy as np
import sys

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
            distance += (abs(points[i][0] - points[i+1][0]) ** 2 +
                         abs(points[i][1] - points[i+1][1]) ** 2) ** 0.5
        return distance

    @staticmethod
    def normalizePoints(points):
        return points - points[0]

    @staticmethod
    def maxDim(points):
        xMin, xMax, yMin, yMax = (sys.maxsize, -sys.maxsize,
                                  sys.maxsize, -sys.maxsize)
        for x, y in points:
            if abs(x) < xMin: xMin = abs(x)
            if abs(x) > xMax: xMax = abs(x)
            if abs(y) < yMin: yMin = abs(y)
            if abs(y) > yMin: yMax = abs(y)
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
        return ((point1[0] - point2[0]) ** 2 +
                (point1[1] - point2[1]) ** 2) ** 0.5

    @staticmethod
    def compareGestures(template, humanGesture):
        def findIndices(templateDistance):
            if templateDistance > template.distanceIndices[-1]:
                return (len(template.distanceIndices) - 2,
                        len(template.distanceIndices) - 1)
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
        return assessment
        # Gesture distance determines number of partitions of the template curve
        # Subsequent distances form indices

    def action(self, *args, **kwargs):
        print "DEFAULT ACTION:", self.name