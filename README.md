# Auto Camera Placement

## Description

I'm an information security student and this is my bachelor's thesis project. The main idea is write a program, 
that can analyze a photo or an image of floorplan and automatically place cameras to protect objects.

All steps of the algorythm here:

1. Process floorplan image with deep neural network for semantic segmentation to get semantic mask with walls,
doorways and windows
2. Break semantic mask into primitive objects (rectangles)
3. Assign a rating to objects according to information security requirements for the location of cameras
4. Perform an algorythm (genetic of brute force) to find a best (at least not worst) cameras placement

Currently second half of overall code here. The first half contains code for deep neural network.
