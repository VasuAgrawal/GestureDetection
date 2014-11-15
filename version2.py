import cv2
import numpy as np

class HandProcessor(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cameraWidth = 1920
        self.cameraHeight = 1080
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.cameraWidth)
        self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.cameraHeight)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

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

    # Should be able to replace this with a "matching" algorithm instead, from here:
    # http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    def findLargestContour(self):
        maxArea, index = 0, 0
        for i in xrange(len(self.contours)):
            area = cv2.contourArea(self.contours[i])
            if area > maxArea:
                maxArea = area
                index = i
        self.largestContour = self.contours[index]
        # self.hullLargestContour = cv2.convexHull(self.largestContour)
        self.hullLargestContour = cv2.convexHull(self.largestContour, returnPoints = False)
        self.defects = cv2.convexityDefects(self.largestContour, self.hullLargestContour)


    def draw(self):
        self.drawingCanvas = np.zeros(self.original.shape, np.uint8)
        self.drawHandContour(True)
        self.drawHullContour(True)
        self.drawDefects(True)
        cv2.imshow('LargestContour', self.drawingCanvas)

    def process(self):
        while (self.cap.isOpened()):
            retVal, self.original = self.cap.read()
            self.original = cv2.flip(self.original, 1)
            self.boostContrast = HandProcessor.boostContrast(self.original)
            self.thresholded = HandProcessor.threshold(self.boostContrast)
            self.setContours(self.thresholded.copy())
            self.findLargestContour()
            self.draw()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.close()

    def getPoint(self, index):
        if index < len(self.largestContour):
            return (self.largestContour[index][0][0], self.largestContour[index][0][1])
        return None

    def drawHandContour(self, bubbles = False):
        cv2.drawContours(self.drawingCanvas, [self.largestContour], 0, (0, 255, 0), 1)
        if bubbles:
            self.drawBubbles(self.largestContour, (255, 255, 0))

    def drawHullContour(self, bubbles = False):
        hullPoints = []
        for i in self.hullLargestContour:
            hullPoints.append(self.largestContour[i[0]])
        hullPoints = np.array(hullPoints, dtype = np.int32)
        cv2.drawContours(self.drawingCanvas, [hullPoints], 0, (0, 0, 255), 2)
        if bubbles:
            self.drawBubbles(hullPoints, (255, 255, 255))

    def drawDefects(self, bubbles = False):
        defectPoints = []
        minDistance = 1000
        for i in self.defects:
            if i[0][3] > minDistance:
                defectPoints.append(self.largestContour[i[0][2]])
        defectPoints = np.array(defectPoints, dtype = np.int32)
        if bubbles:
            self.drawBubbles(defectPoints, (0, 0, 255), width = 4)

    def drawBubbles(self, pointsList, color = (255, 255, 255), width = 2):
        for i in xrange(len(pointsList)):
            for j in xrange(len(pointsList[i])):
                cv2.circle(self.drawingCanvas, (pointsList[i][j][0], pointsList[i][j][1]), width, color)

HandProcessor().process()

class HandProcessorSingleImage(HandProcessor):
    def __init__(self):
        self.original = cv2.imread('oneHand.jpg')

    def process(self):
        self.boostContrast = HandProcessor.boostContrast(self.original)
        self.thresholded = HandProcessor.threshold(self.boostContrast)
        self.setContours(self.thresholded.copy())
        self.findLargestContour()
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
        cv2.imshow('LargestContour', self.drawingCanvas)

# HandProcessorSingleImage().process()