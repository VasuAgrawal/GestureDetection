from eventBasedAnimationClass import EventBasedAnimationClass
from Tkinter import *
from GesturesApi import GestureProcessor
from PIL import Image, ImageTk
# http://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter


class GestureDemo(EventBasedAnimationClass):
    def __init__(self):
        self.gp = GestureProcessor("Somefile.txt") # will default to usual file
        super(GestureDemo, self).__init__(width=1920, height=1080)
        # super(GestureDemo, self).__init__(width=self.gp.cameraWidth,
        #                                     height=self.gp.cameraHeight)
        self.timerDelay = 1000 / 30 # 30 FPS
        self.bindGestures()

    def bindGestures(self):
        def action(self):
            print "LOOK I'M WORKING FROM TKINTER,", self.name
        for gName in self.gp.getGestureNames():
            self.gp.bind(gName, action)

    def onMousePressed(self, event):
        print "You could probably do something with the coords:", (event.x, event.y)

    def onKeyPressed(self, event):
        if event.char == 'r':
            self.gp.saveNext()
        elif event.char == 'q':
            self.onClose()
            exit()

    def onTimerFired(self):
        self.gp.process()

    def redrawAll(self):
        cv2image = self.gp.getRGBAOriginal()
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
        self.imagetk = imgtk # Need this for persistence because of garbage collection
        self.canvas.create_image(0, 0, image=imgtk, anchor="nw")

        self.gp.draw()
        cv2image = self.gp.getRGBACanvas()
        imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
        self.imagetk2 = imgtk2 # Need this for persistence because of garbage collection
        self.canvas.create_image(1920, 1080, image=imgtk2, anchor="se")

    def run(self):
        super(GestureDemo, self).run()
        self.onClose()

    def onClose(self):
        self.gp.close() # MUST DO THIS

GestureDemo().run()