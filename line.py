import numpy as np

class Line(object):
        inf = np.inf

        @staticmethod
        def distance(point1, point2):
            return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

        def __init__(self, point1, point2):
            self.point1 = point1
            self.point2 = point2
            self.slope = self.getSlope()
            self.midpoint = np.array([(point1[0] + point2[0]) / 2.0, (point1[1] + point2[1]) / 2.0])
            if self.slope == Line.inf:
                self.pslope = 0
            elif self.slope == 0:
                self.pslope = Line.inf
            else:
                self.pslope = -1/self.slope

        def getSlope(self):
            den = (self.point2[0] - self.point1[0])
            if den != 0:
                return float(self.point2[1] - self.point1[1]) / den
            else:
                return Line.inf

        @staticmethod
        def colinear(points):
            mpoints = [tuple(point) for point in points]
            mpoints.sort()
            if Line(mpoints[1], mpoints[2]).slope == Line(mpoints[0], mpoints[1]).slope == Line(mpoints[0], mpoints[2]).slope:
                return True
            return False

        def __str__(self):
            return "Point1: %s Point2: %s Slope: %s PSlope: %s Midpoint: %s" % (self.point1, self.point2, self.slope, self.pslope, self.midpoint)