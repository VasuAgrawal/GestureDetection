import cv2
import numpy as np

class bgSub(object):
    def __init__(self):
       self.cap = cv2.VideoCapture(0)
       self.cameraWidth = 1920
       self.cameraHeight = 1080
       self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.cameraWidth)
       self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.cameraHeight)
       self.bg = None
       self.toSub = None
       self.frameCount = 0

    def process(self):
        while (True):
            retval, self.original = self.cap.read()
            self.frameCount += 1
            # self.modifyBG()
            self.denoise()
            cv2.imshow('Original', self.original)
            # cv2.imshow('BG', self.bg)
            cv2.imshow('Denoised', self.denoised)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if self.frameCount == 2:
            #     break
        self.close()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def denoise(self):
        self.denoised = np.array(self.original.shape, dtype = np.uint8)
        cv2.fastNlMeansDenoisingColored(self.original, self.denoised)

    def modifyBG(self):
        if self.bg == None:
            self.bg = self.original.copy()
        else:
            # Need to do this with numpy arrays to avoid speed issues
            # floored subtraction
            # subtracted = self.original - self.bg
            # mask = subtracted > self.original
            # mask = np.invert(mask)
            # subtracted = subtracted * mask
            sub = self.original - self.bg
            rsub = self.bg - self.original
            tol = 20
            # print sub
            # print rsub
            mask = (sub < tol) & (rsub < tol)
            # print mask.shape
            mask = mask.max(2)
            # print mask.shape
            mask = np.dstack((mask, mask, mask))
            self.bg = self.original * mask
            # print mask

    @staticmethod
    def weightedAverage(old, new, weight, minWeight, equal = False):
        if equal:
            return (self + old) / 2
        else:
            weighted = None
            if 1/float(weight) < minWeight:
                weighted = old * (1-minWeight) + new * minWeight
            else:
                weighted = (old * weight + new) / (weight + 1)
            return weighted

# print bgSub.weightedAverage(np.array([1,1,1]), np.array([5, 5, 25]), 1, 1/15.0)

bgSub().process()