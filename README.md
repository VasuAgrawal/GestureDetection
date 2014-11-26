Basic Gesture Detection
===========

This program uses Python 2 and OpenCV 2 to attempt to detect the user's hand gestures. Gestures can be mapped to different actions and new gestures can be trained.

Currently works best in low lighting with the hand as the brightest portion of the image.

Features:

* Detect a single hand and obtain contour
* Determine center of the detected hand
* Track the hand over multiple frames
* Detect possible gestures
* Basic classification of possible gestures
* Record mode (new gesture training)
* Hookable API
* Sample code

Yet to be implemented (in no particular order):

* Better hand detection
* Gesture training (improving old gestures)
* Finger detection
* Depth perception

Installation
===========

Linux: Required modules are probably bundled with your favorite distribution of Linux (i.e. Ubuntu, Debian, Linux Mint, etc.) However, in the event that it needs to be installed, the following instructions can be followed:

1. On Debian/Ubuntu/similar systems: `sudo apt-get install python`
2. Download and install OpenCV 2 for Python: http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html

Usage
===========

See `testImplementation.py` for example. More documentation to come. (Running any of the other files won't do anything.)