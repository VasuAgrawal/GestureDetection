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
        # def fn(): print "HELLO IM YELLOW"
        # self.gp.gestures[0].action = fn
        # def fn(): print "EHLLO I'm PURPLSE"
        # self.gp.bind(0, fn)

    def bindGestures(self):
        def fn(): print "HELLO IM YELLOW"
        self.gp.gestures[0].action = fn
        def fn(): print "EHLLO I'm PURPLSE"
        self.gp.bind(0, fn)
        # print self.gp.getGestureNames()
        # def action():
        #     return "LOOK I'M WORKING FROM TKINTER"
        # for i in xrange(len(self.gp.gestures)):
        #     def fn():
        #         print action() + self.gp.gestures[i].name
        #     self.gp.bind(i, fn)

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

    # This is terrible code and I should really come up with a better way of doing this
    def redrawAll(self):
        self.canvas.delete(ALL)
        cv2image = self.gp.getRGBAOriginal()
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
        self.imagetk = imgtk # Need this for persistence because of garbage collection
        self.canvas.create_image(0, 0, image=imgtk, anchor="nw")

        cv2image = self.gp.getRGBAThresh(.75, .75)
        imgtk3 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
        self.imagetk3 = imgtk3 # Need this for persistence because of garbage collection
        self.canvas.create_image(0, 1080, image=imgtk3, anchor="sw")

        self.gp.draw()
        cv2image = self.gp.getRGBACanvas()
        imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
        self.imagetk2 = imgtk2 # Need this for persistence because of garbage collection
        self.canvas.create_image(1920, 1080, image=imgtk2, anchor="se")

        self.canvas.create_text(1920, 0, text=self.gp.lastAction, anchor="ne", font = "15")

    def run(self):
        super(GestureDemo, self).run()
        self.onClose()

    def onClose(self):
        self.gp.close() # MUST DO THIS

GestureDemo().run()