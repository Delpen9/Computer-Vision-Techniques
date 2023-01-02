# Problem Set 5: Object Tracking and Pedestrian Detection

## NOTE: Please refer to the PDF file for instructions. GitHub markdown does not render formulae and algorithms correctly. The information is identical between the markdown and PDF, but the PDF renders more cleanly.

# Assignment Description

## Description
In Problem set 5, you are going to implement tracking methods for image sequences and videos.
The main algorithms you will be using are the Kalman and Particle Filters.
Methods to be used:You will design and implement a Kalman and a Particle filters from the ground up.
RULES: You may use image processing functions to find color channels, load images, find edges(such as with Canny), resize images.  Don’t forget that those have a variety of parameters andyou may need to experiment with them.
There are certain functions that may not be allowedand are specified in the assignment’s autograder Piazza post.Refer to this problem set’s autograder post for a list of banned function calls.Please do not use absolute paths in your submission code.
All paths should be relative to thesubmission directory. Any submissions with absolute paths are in danger of receiving a penalty!


## Learning Objectives

 - Identify  which  image  processing  methods  work  best  in  order  to  locate  an  object  in  ascene.
 - Learn to how object tracking in images works.
 - Explore different methods to build a tracking algorithm that relies on measurements anda prior state.
 - Create methods that can track an object when occlusions are present in the scene.

