Basic Gesture Detection
===========

Demo: https://www.youtube.com/watch?v=oH0ZkfFoeYU

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

1. On Debian/Ubuntu/etc: `sudo apt-get install python`
2. Download and install NumPy: `sudo apt-get install python-numpy`
3. Download and install OpenCV 2 for Python: http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html

Usage
===========

See `testImplementation.py` for example. 

1. Download these files and place all of the files from the `current_src` directory into your project's working directory (to allow easy importing of modules)
2. The processor is provided as an object `GestureProcessor` in the GesturesApi file. To use, simply add the following import statement: `from GesturesApi import GestureProcessor`
3. Create an instance of `GestureProcessor`, such as `gp = GestureProcessor()`. You can optionally pass it a `.txt` file which contains coordinates defining a gesture. If no file is specified, default gestures will be loaded (generation code can be found in `defaultGesturesLoader.py`). Note: No checking is done for data integrity; it is assumed that the provided file meets the proper format specifications.
4. Bind gestures as desired, using the `bind()` method: `gp.bind(index, fn)`. `index` can either be the integer index of the gesture (in the order that the gestures were loaded) or a string containing the exact name of the gesture. `fn` is a function object which takes no parameters; use closures as necessary (i.e. `gp.bind(index, lambda: self.fn)`)
5. In the main loop, call `gp.process()`. This will grab the next camera image and update the information inside `gp`, including depth and palm center. If a gesture is detected, this will also call the action that was bound to it, and update `gp.lastAction` with the name of the last gesture. Note: this call is expensive and will take anywhere between 2 to 5 ms on average, depending on the machine.
6. You can record new gestures by calling `gp.saveNext()`. This will add the next new gesture to the list of gestures with a random name. The random name is then set as `gp.lastAction`, so the programmer can change it to something more useful if desired. Alternatively, the gesture will simply be the last one in `gp.gestures`, and can be modified from there.
7. Upon exit of the program, it is CRITICAL to call `gp.close()`. This will clean up created data and, importantly, release the camera. Failure to do so will result in the camera being active after the program appears to have exited, and will make it impossible for other applications to bind onto the camera (including new instances of the offending program.)

Algorithm
===

The `process()` loop can be summarized as follows:

1. Read from camera
2. Convert image to binary color using thresholding
3. Use OpenCV to extract contours from image
4. Find the largest contour from detected list, which is assumed to be the hand contour
5. Use the bounding box to determine the width and height of the contour
6. Use OpenCV to find the convex hull and convexity defects of the contour
7. Find the center of the hand by looking for the largest possible inscribed circle
8. Use the radius of the inscribed circle to approximate a distance
9. Use the positions of the palm center between times when the hand is stationary to determine gestures
10. Compare the list of points against all the template gestures:
  1. Find the total distance of the tracked and template gestures
  2. Determine how far along the total distance each point falls
  3. For each point in the tracked gesture, look for the two template points which are closest to the fraction of the distance that the tracked point is
  4. Linearize between the two template points to find a point to compare the tracked point to
  5. Keep a running tally of the distance difference
  6. Find which of the gestures has the lowest distance and see if it is lower than a reasonable threshold
11. Initiate callback function if tracked gestures has matched a template
