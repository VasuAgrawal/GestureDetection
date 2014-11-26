import numpy as np
import cv2

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.BackgroundSubtractorGMG()
fgbg = cv2.BackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()