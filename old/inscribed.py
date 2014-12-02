def setPalmCombinations(self):
    # Testing confirms that the range of distances is about 5 - 10 for
    # minimum distance between points, and anywhere from 150 - 200 for max
    # distance between points. 10 and 100 should be reasonable cutoffs.
    minDistance = 10
    maxDistance = 100
    iter = itertools.combinations(self.handContour, 3)
    def distanceCheck(points):
        A = points[0][0]
        B = points[1][0]
        C = points[2][0]
        AB = np.linalg.norm(A - B)
        if not(minDistance < AB < maxDistance): return None
        BC = np.linalg.norm(B - C)
        if not(minDistance < BC < maxDistance): return None
        CA = np.linalg.norm(C - A)
        if not(minDistance < CA < maxDistance): return None
        return np.array([A, B, C])
    a = [distanceCheck(i) for i in iter]

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