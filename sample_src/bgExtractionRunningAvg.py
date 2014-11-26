# http://opencvpython.blogspot.in/2012/07/background-extraction-using-running.html

import cv2
import numpy as np
 
c = cv2.VideoCapture(0)
_,f = c.read()
 
avg1 = np.float32(f)
avg2 = np.float32(f)
 
while(1):
    _,f = c.read()
     
    cv2.accumulateWeighted(f,avg1,0.1)
    cv2.accumulateWeighted(f,avg2,0.01)
        
    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    subtracted = cv2.blur(f, (5,5)) - res1

    cv2.imshow('sub', subtracted)

    subtracted = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
    retval, thresh = cv2.threshold(subtracted, 50, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.erode(thresh, (7, 7))
    thresh = cv2.dilate(thresh, (7, 7))

    # grey = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
    # value = (31, 31)
    # blurred = cv2.GaussianBlur(grey, value, 0)
    # retVal, thresh = cv2.threshold(blurred, 0, 255,
    #                                 cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow('img',f)
    cv2.imshow('avg1',res1)
    cv2.imshow('avg2',res2)
    cv2.imshow('thresh', thresh)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()
c.release()