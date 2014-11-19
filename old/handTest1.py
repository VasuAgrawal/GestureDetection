import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 60)

while ( cap.isOpened()):
    ret, img = cap.read()

    img = cv2.flip(img, 1)

    cv2.imshow('original', img)
    # b, g, r = cv2.split(img)
    # cv2.imshow('blue channel', b)
    # cv2.imshow('red channel', r)
    # cv2.imshow('green channel', g)

    maxIntensity = 255.0
    phi = 1
    theta = 1
    contrastBoost = (maxIntensity/phi) * (img/(maxIntensity/theta)) ** 2
    contrastBoost = np.array(contrastBoost, np.uint8)

    cv2.imshow('contrastBoost', contrastBoost)


    grey = cv2.cvtColor(contrastBoost, cv2.COLOR_BGR2GRAY)
    grey2 = grey.copy()
    # print cv2.Canny(grey, 100, 200)
    # break
    cv2.imshow('canny?', grey2)

    blur = cv2.GaussianBlur(grey, (31, 31), 0)

    # ret, thresh = cv2.threshold(blur, 118, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      
    cv2.imshow('threholded', thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area, ci = 0, 0
    for i in xrange(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            prev_max_area = max_area
            prev_ci = ci
            max_area = area
            ci = i
    print max_area
    cnt = contours[ci]
    hull = cv2.convexHull(cnt)

    # print cv2.minAreaRect(cnt)
    # break


    # minRectHull, junk = cv2.minAreaRect(hull)

    # for i in xrange(len(hull)):
    #     for j in xrange(len(hull[i])):
    #         print hull[i][j][0], hull[i][j][1]

    # print
    # print

    drawing = np.zeros(img.shape, np.uint8)

    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), -1)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

    # cv2.drawContours()
    
    # cv2.drawContours(drawing, [minRect], 0, (0, 255, 255), 2)
    # cv2.drawContours(drawing, [minRectHull], 0, (255, 0, 255), 2)




    cv2.imshow('drawing', drawing)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
