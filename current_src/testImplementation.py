from eventBasedAnimationClass import EventBasedAnimationClass
from Tkinter import *
from GesturesApi import GestureProcessor
from PIL import Image, ImageTk
# Import statements from: 
# http://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter

class Smiley(object):
    def __init__(self, x, y, radius=50, image=None):
        self.x = x
        self.y = y
        self.radius = radius
        self.handles = []
        self.image = image
        self.clearImage = True
        self.initImage()

    # Adapted from the following link:
    # http://stackoverflow.com/questions/4066202/
    # resizing-pictures-in-pil-in-tkinter
    def initImage(self):
        if self.image != None:
            self.image = Image.open(self.image)
            self.image = self.image.resize((self.radius * 2, self.radius * 2),
                                           Image.ANTIALIAS)
            self.imageTk = ImageTk.PhotoImage(self.image)

    def drawSmiley(self, canvas):
        if self.clearImage:
            self.delete(canvas)
        if self.image == None:
            self.handles.append(canvas.create_oval(self.x - self.radius,
                                self.y - self.radius, self.x + self.radius,
                                self.y + self.radius, fill="yellow"))
            eyeRad = self.radius / 5
            self.handles.append(canvas.create_oval(self.x - self.radius / 2,
                                self.y - self.radius / 2,
                                self.x - self.radius / 2 + eyeRad,
                                self.y - self.radius / 2 + eyeRad,
                                fill="black"))
            self.handles.append(canvas.create_oval(
                                self.x + self.radius / 2 - eyeRad,
                                self.y - self.radius / 2,
                                self.x + self.radius / 2,
                                self.y - self.radius / 2 + eyeRad,
                                fill="black"))
            self.handles.append(canvas.create_arc(self.x - self.radius / 2,
                                self.y - self.radius / 2,
                                self.x + self.radius / 2,
                                self.y + self.radius / 2,
                                start=200, extent=140, style=ARC,
                                width=self.radius / 10))
        else:
            self.handles.append(canvas.create_image(self.x, self.y,
                                image=self.imageTk, anchor="center"))

    def delete(self, canvas):
        for handle in self.handles:
            canvas.delete(handle)
        self.handles = []


# Subclasses Object from here:
# http://www.cs.cmu.edu/~112/notes/eventBasedAnimationClass.py
class GestureDemo(EventBasedAnimationClass):
    def __init__(self):
        self.gp = GestureProcessor("Gesture_data.txt")  # default to usual file
        self.width = 1920
        self.height = 1080
        super(GestureDemo, self).__init__(width=self.width, height=self.height)
        self.timerDelay = 1000 / 30 # 30 FPS
        # self.bindGestures()
        self.CVHandles = []
        self.bgHandle = None
        self.trackCenter = False
        self.showSmiley = False
        self.showLukas = False
        self.trail = False

    def initAnimation(self):
        self.smiley = Smiley(self.width * 3 / 4, self.height / 4)
        self.lukas = Smiley(self.width * 3 / 4, self.height / 4,
                            image="lbp.jpg")
        self.drawBG()

    def drawSmiley(self):
        self.showSmiley = True
        self.showLukas = False
        self.lukas.delete(self.canvas)

    def drawLukas(self):
        self.showLukas = True
        self.showSmiley = False
        self.smiley.delete(self.canvas)

    def bindGestures(self):
        self.gp.bind("Infinity", lambda: self.drawLukas())
        self.gp.bind("Diagonal Bottom Left to Top Right",
                     lambda: self.drawSmiley())

    def bindHandlers(self):
        self.root.bind("<KeyPress>", lambda event: self.onKeyDown(event))
        self.root.bind("<KeyRelease>", lambda event: self.onKeyUp(event))

    def onMousePressed(self, event):
        print "Mouse Clicked at:", (event.x, event.y)

    def onKeyPressed(self, event):
        if event.char == 'r':
            self.gp.saveNext()
        elif event.char == 's':
            self.trackCenter = not self.trackCenter
        elif event.char == 'c':
            self.trail = not self.trail
        elif event.char == 'd':
            self.canvas.delete(ALL)
            self.drawBG()
        elif event.char == 'b':
            self.bindGestures()
        elif event.char == 'q':
            self.onClose()
            exit()

    def onTimerFired(self):
        self.gp.process()
        self.updateSmiley()

    def updateSmiley(self):
        if self.trackCenter:
            self.smiley.x = int((self.gp.getScaledCenter()[0] + 1) *
                                (self.width / 2))
            self.smiley.y = int((self.gp.getScaledCenter()[1]) *
                                (self.height / 2))
            self.smiley.radius = int(self.gp.handDistance) * 2
            self.lukas.x = int((self.gp.getScaledCenter()[0] + 1) *
                                (self.width / 2))
            self.lukas.y = int((self.gp.getScaledCenter()[1]) *
                                (self.height / 2))
        if self.trail:
            self.smiley.clearImage = False
            self.lukas.clearImage = False
        else:
            self.lukas.clearImage = True
            self.smiley.clearImage = True

    # OpenCV Image drawing adapted from:
    # http://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter
    def drawCVImages(self):
        for handle in self.CVHandles:
            self.canvas.delete(handle)
        self.CVHandles = []

        cv2image = GestureProcessor.getRGBAFromBGR(self.gp.original,
                                                   self.width / 2,
                                                   self.height / 2)
        self.imagetk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        self.gp.draw()
        cv2image = GestureProcessor.getRGBAFromBGR(self.gp.drawingCanvas,
                                                   self.width / 2,
                                                   self.height / 2)
        self.imagetk2 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        cv2image = GestureProcessor.getRGBAFromGray(self.gp.thresholded,
                                                    self.width / 2,
                                                    self.height / 2)
        self.imagetk3 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        self.CVHandles.append(self.canvas.create_image(0, 0, image=self.imagetk,
                              anchor="nw"))
        self.CVHandles.append(self.canvas.create_image(1920, 1080,
                              image=self.imagetk2, anchor="se"))
        self.CVHandles.append(self.canvas.create_image(0, 1080,
                              image=self.imagetk3, anchor="sw"))

        self.CVHandles.append(self.canvas.create_text(1920, 0,
                              text=self.gp.lastAction, anchor="ne",
                              font="15"))
        self.CVHandles.append(self.canvas.create_text(1920, 20,
                              text="Distance: " + str(round(
                                                      self.gp.handDistance, 3)),
                              anchor="ne", font="15"))
        self.CVHandles.append(self.canvas.create_text(1920, 40,
                              text=str(self.gp.getScaledCenter()),
                              anchor="ne", font="15"))

    def drawBG(self):
        self.bgHandle = self.canvas.create_rectangle(self.width/2, 0,
                                                     self.width, self.height/2,
                                                     fill="white")

    def redrawAll(self):
        self.drawCVImages()
        if self.showSmiley:
            self.smiley.drawSmiley(self.canvas)
        elif self.showLukas:
            self.lukas.drawSmiley(self.canvas)

    def run(self):
        super(GestureDemo, self).run()
        self.onClose()

    def onClose(self):
        self.gp.close()  # MUST DO THIS

GestureDemo().run()
