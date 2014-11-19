# import numpy as np
# import cv2

# img = np.zeros((300, 300, 3), np.uint8)

# cv2.line(img, (0,0), (299, 299), (255, 0, 0), 4)
# cv2.rectangle(img, (150, 150), (200, 200), (255, 255, 255), 10)
# cv2.circle(img, (175, 175), 25, (255, 0, 0), 3)

# points = np.array([[0, 250], [0, 200], [15, 230], [20, 240]], np.int32)
# points = points.reshape((-1, 1, 2))
# cv2.polylines(img, [points], True, (0, 255, 0))

# cv2.imshow('LOOKATME', img)
# cv2.waitKey(0)

# import cv2
# print [i for i in dir(cv2) if 'EVENT' in i]

# import cv2
# import numpy as np

# drawing = False # true if mouse is pressed
# mode = True # if True, draw rectangle. Press 'm' to toggle to curve
# ix,iy = -1,-1

# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#             else:
#                 cv2.circle(img,(x,y),5,(0,0,255),-1)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv2.circle(img,(x,y),5,(0,0,255),-1)

# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)

# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break

# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# def nothing(x):
#     pass

# # Create a black image, a window
# img = np.zeros((300,512,3), np.uint8)
# cv2.namedWindow('image')

# # create trackbars for color change
# cv2.createTrackbar('R','image',0,255,nothing)
# cv2.createTrackbar('G','image',0,255,nothing)
# cv2.createTrackbar('B','image',0,245,nothing)

# # create switch for ON/OFF functionality
# switch = '0 : OFF \n1 : ON'
# cv2.createTrackbar(switch, 'image',0,1,nothing)

# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

#     # get current positions of four trackbars
#     r = cv2.getTrackbarPos('R','image')
#     g = cv2.getTrackbarPos('G','image')
#     b = cv2.getTrackbarPos('B','image')
#     s = cv2.getTrackbarPos(switch,'image')

#     if s == 0:
#         img[:] = 0
#     else:
#         img[:] = [b,g,r]

# cv2.destroyAllWindows()

prevMouse = (0, 0)

def onMouseHandle(event, x, y, flags, param):
    global prevMouse
    if (x,y) != prevMouse:
        print "MousePos: %d, %d" % (x,y)
    prevMouse = (x,y)


import cv2
import numpy as np

img = cv2.imread('test.jpg', 1)

boobies = img[150:480, 505:830]
img[150:480, 275:430] = boobies

cv2.namedWindow('image')
cv2.setMouseCallback('image', onMouseHandle)

while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        cv2.imshow('image', img)

cv2.destroyAllWindows()