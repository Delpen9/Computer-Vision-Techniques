# Problem Set 6: Image Classification

## Assignment Description

In this problem set you will be implementing face recognition using PCA,
(Ada)Boosting, and the Viola-Jones algorithm.

## Learning Objectives

-   Learn the advantages and disadvantages of PCA, Boosting, and
    Viola-Jones.

-   Learn how face detection/recognition work, explore the pros & cons
    of various techniques applied to this problem area.

-   Identify the inherent challenges of working on these face detection
    methods.

## Problem Overview

### Methods to be used

In this assignment you will be implementing PCA, Boosting, and
HaarFeatures algorithms from scratch. Unlike previous problem sets, you
will be coding them without using OpenCV functions dedicated to solve
the problem.
**Please do not use absolute paths in your submission code. All paths
should be relative to the submission directory. Any submissions with
absolute paths are in danger of receiving a penalty!**

# Instructions

## Programming Instructions 

Your main programming task is to complete the api described in the file
**ps6.py**. The driver program **experiment.py** helps to illustrate the
intended use and will output the files needed for the writeup.
Additionally there is a file **ps6_test.py** that you can use to test
your implementation. **Note** for this assignment, the [numba
library](https://pypi.org/project/numba/) is required for efficient code
runtime. The library is just included for the helper classes and you
will not be required to learn the library to use it. Simply use 'pip
install numba' to install it. **Please merge that input_images_part1 and part2 
into one folder named input_images. Else the starter code will not work as is.**

## Write-up Instructions 

Create **ps6_report.pdf** - a PDF file that shows all your output for
the problem set, including images labeled appropriately (by filename,
e.g. ps6-1-a-1.png) so it is clear which section they are for and the
small number of written responses necessary to answer some of the
questions (as indicated). For a guide as to how to showcase your
results, please refer to the latex template for PS4.

## How to Submit 

Two assignments have been created on Gradescope: one for the report -
PS6_report, and the other for the code - **PS6_code**.

-   Report: the report (PDF only) must be submitted to the PS6_report
    assignment.

-   Code: all files must be submitted to the PS6_code assignment. DO NOT
    upload zipped folders or any sub-folders, please upload each file
    individually. Drag and drop all files into Gradescope.

### Notes 

-   You can only submit to the autograder **10** times in an hour.
    You'll receive a message like \"You have exceeded the number of
    submissions in the last hour. Please wait for 36.0 mins before you
    submit again.\" when you exceed those 10 submissions. You'll also
    receive a message \"You can submit 8 times in the next 53.0 mins\"
    with each submission so that you may keep track of your submissions.

-   If you wish to modify the autograder functions, create a copy of
    those functions and DO NOT mess with the original function call.

**YOU MUST SUBMIT your report and code separately, i.e., two submissions
for the code and the report, respectively. Only your last submission
before the deadline will be counted for each of the code and the
report.**

# PCA \[30 points\]

Principal component analysis (PCA) is a technique that converts a set of
attributes into a smaller set of attributes, thus reducing
dimensionality. Applying PCA to a dataset of face images generates
eigenvalues and eigenvectors, in this context called eigenfaces. The
generated eigenfaces can be used to represent any of the original faces.
By only keeping the eigenfaces with the largest eigenvalues (and
accordingly reducing the dimensionality) we can quickly perform face
recognition.
In this part we will use PCA to identify people in face images. We will
use the Yalefaces dataset, and will try to recognize each individual
person.

## Loading images

Loading images: As part of learning how to process this type of input,
you will complete the function load_images. Read each image in the
images_files variable, resize it to the dimensions provided, and create
a data structure X that contains all resized images. Each row of this
array is a flattened image (see np.flatten).\
You will also need the labels each image belongs to. Each image file has
been named using the following format:subject##.xyz.png. We will use the
number in ## as our label ("01" -\> 1, "02" -\> 2, etc.).\
Create a list of labels that match each image in X using the filename
strings. Next, use the X array to obtain the mean face $(\mu)$ by
averaging each column. You will then use the resulting structure to
reshape it to a 2D array and later save it as an image. Complete
visualize_mean_face in experiment.py to produce images for the report.

**Code:**
-load_images(folder, size=(32,32))
-get_mean_face(x)
-visualize_mean_face(x_mean, size, new_dims)
**Report:** Mean face image: **ps6-1-a-1.png**

## PCA

Now that we have the data points in X and the mean $\mu$, we can go
ahead and calculate the eigenvectors and eigenvalues to determine the
vectors with largest covariance. See [Eigenfaces for
Recognition](http://www.cs.ucsb.edu/~mturk/Papers/jcn.pdf) for more
details. Using the equation from the lectures:\
You need to find the eigenvectors $\mu$. Luckily there is a function in
Numpy linalg.eigh that can do this using $\sum$ as an input.
$$\mu^T \Sigma \mu$$ where
$$\Sigma = \frac{1}{M} \sum\limits_{i=1}^N (x_i - \mu)(x_i - \mu)^T$$

**Code:** pca(X, k))
**Report:** Top 10 eigenfaces: ps6-b-1.png

## Face Recognition (classification)

Now that we have the PCA reduction method ready, let's continue with a
simple (naive) classification method.
First, we need to split the data into a training and a test set. You
will perform this operation in the split_dataset function. Next, the
training stage is defined as obtaining the eigenvectors from using the
training data. Each image face in the training and test set is then
projected to the "face space" using these vectors.
Finally, find the eigenface in the training set that is closest to each
projected face in the test set. We will use the label that belongs to
the training eigenface as our predicted class. Your task here is to do a
quick analysis of these results and include them in your report.
**Code:** split_dataset(X, y, p))
**Report:** Analyze the accuracy results over multiple iterations. Do
these "predictions" perform better than randomly selecting a label
between 1 and 15? Are there any changes in accuracy if you try low
values of k? How about high values? Does this algorithm improve changing
the split percentage p?

# Boosting \[20 points\]

In this part we will classify images of fruits \[apple or pear\] created
by [Horea et al](https://github.com/Horea94/Fruit-Images-Dataset). You
may find the contents on boosting in [page 663 of Szeliski's (2010)
book](http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf)
useful for this section. Recall from the lectures that the idea behind
boosting is to use a combination of "weak classifiers". Each weak
classifier's vote is weighted based on its accuracy.
The Boosting class will contain the methods behind this algorithm. You
will use the class WeakClassifier as a classifier ($h(x)$) which is
implemented to predict based on threshold values. Boosting creates a
classifier $H(x)$ which is the combination of simple weak classifiers.
Complete the Boosting.train() function with the Adaboost algorithm
(modified for this problem set):

Please see instructions.pdf for detailed algrithm.

## 

Using the Fruits dataset, split the dataset into training ($X_{train}$)
and testing ($X_{test}$) data with their respective labels ($y_{train}$)
and ($y_{test}$). The naming of the files follows the other datasets so
use the load_images() function from the previous section. Perform the
following tasks:

1.  Establish a baseline to see how your classifier performs. Create
    predicted labels by selecting N random numbers $\in \{−1,1\}$ where
    N is the number of training images. Report this method's accuracy
    (as a percentage): $100 * \frac{correct}{total}$.

2.  Train WeakClassifier using the training data and report its accuracy
    percentage

3.  Train Boosting.train() for num_iterations

4.  Train Boosting.train() for num_iterations and report the training
    accuracy percentage by calling Boosting.evaluate().

5.  Do the same for the testing data. Create predicted labels by
    selecting N random numbers $\in \{−1,1\}$ where N is the number of
    testing images. Report its accuracy percentage.

6.  Use the trained WeakClassifier to predict the testing data and
    report its accuracy.

7.  Use the trained BoostingClassifier to predict the testing data and
    report its accuracy.

**Code:**

-   ps6.Boosting.train()

-   ps6.Boosting.predict(X)

-   ps6.Boosting.evaluate()

**Report:**

-   Text Answer: Report the average accuracy over 5 iterations. In each
    iteration, load and split the dataset, instantiate a Boosting object
    and obtain its accuracy.

-   Text Answer: Analyze your results. How do the Random, Weak
    Classifier, and Boosting perform? Is there any improvement when
    using Boosting? How do your results change when selecting different
    values for num_iterations? Does it matter the percentage of data you
    select for training and testing (explain your answers showing how
    each accuracy changes).

# Haar-like Features \[20 points\]

In this section you will work with Haar-like features which are normally
used in image classifications. These can act to encode ad-hoc domain
knowledge that is difficult to learn using just pixel data. We will be
using five types of features: two-horizontal, two-vertical,
three-horizontal, three-vertical, and four-rectangle. You may want to
review the contents in Lesson 8C-L2 and the [Viola-Jones
paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-IJCV-01.pdf).
In this section you will complete the HaarFeatureclass.

## 

You will start by generating grayscale image arrays that contain these
features. Create the arrays based on the parameters passed when
instantiating a HaarFeatureobject. These parameters are:\
-type: sets the typeof feature among two_horizontal, two_vertical,
three_horizontal, three_vertical, and four_square (see examples below).\
-position: represents the top left corner (row, col) of the feature in
the image.\
-size: (height, width) of the area the feature will occupy.\
Complete the function HaarFeatures.preview(). You will return an array
that represents the Haar features, much like each of the five shown
above. These Haar feature arrays should be based on the parameters used
to instantiate each Haar feature. Notice that, for visualization
purposes, the background must be black (0), the area of addition white
(255), and the area of subtraction gray (126). Note that the area
occupied by a feature should be evenly split into its component areas
-three-horizontal should be split into 3 evenly sized areas, four-square
should be split into 4 evenly sized areas (in other words, divide the
width and height evenly using int division).\
**Code:**

-   Functions in HaarFeatures.preview()

**Report:**

-   Input: type = two_horizontal; position = (25, 30); size = (50, 100)
    Output: ps6-3-a-1.png

-   Input: type = two_vertical; position = (10, 25); size = (50, 150)
    Output: ps6-3-a-2.png

-   Input: type = three_horizontal; position = (50, 50); size =
    (100, 50) Output: ps6-3-a-3.png

-   Input: type = three_vertical; position = (50, 125); size = (100, 50)
    Output: ps6-3-a-4.png

-   Input: type = four_square; position = (50, 25); size = (100, 150)
    Output: ps6-3-a-5.png

# Viola-Jones \[30 Points\]

In this part we will use Haar-like features and apply them in face
recognition. Haar-like features can act to encode ad-hoc domain
knowledge that is difficult to learn using just pixel data. We will be
using five types of features: two-horizontal, two-vertical,
three-horizontal, three-vertical, and four-rectangle. You may want to
review the contents in Lesson 8C-L2 and the [Viola-Jones
paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-IJCV-01.pdf).\
Haar-like features can be used to train classifiers restricted to using
a single feature. The results from this process can be then applied in
the boosting algorithm explained in the Viola-Jones paper. We will use a
dataset of images that contain faces (pos/ directory) and refer to them
as positive examples. Additionally, use the dataset of images in the
neg/ directory which contain images of other objects and refer to them
as negative examples. Code the boosting algorithm in the ViolaJones
class.\
First, we will be using images resized to 24x24 pixels. This set of
images are converted to integral images using the function you coded
above. Instantiate a ViolaJones object using the positive images,
negative images, and the integral images. ViolaJones(train_pos,
train_neg, integral_images). Notice that the class contains one
attribute to store Haar Features (self.haarFeatures) and another one for
classifiers (self.classifiers).\
We have provided a function createHaarFeaturesthat generates a large
amount of features within a 24x24 window. This is why it is important to
use integral images in this process.\
Use the Boosting algorithm in the Viola-Jones paper (labeled Table 1) as
a reference. You can find a summary of it below adapted to this problem
set.\

Please see instructions.pdf for the algorithm

## 

Complete ViolaJones.train(num_classifiers)with the algorithm shown
above. We have provided an initial step that generates a 2D array with
all the feature scores per image. Use this when you instantiate a
VJ_Classifier along with the labels and weights. After calling the weak
classifier's train function you can obtain its lowest error $\epsilon$
by using the 'error' attribute.
**Code:** ViolaJones.train(num_classifiers)

## 

Complete ViolaJones.predict(images)implementing the strong classifier
$H(x)$ definition.

**Code:** ViolaJones.predict(images)
**Report:**
-Output: ps6-4-b-1.png and ps6-4-b-2.png which correspond to the first
two Haar features selected during the training process.\
-Text Answer: Report the classifier accuracy both the training and test
sets with a number of classifiers set to 5. What do the selected Haar
features mean? How do they contribute in identifying faces in an image?

## 

Now that the ViolaJones class is complete, you will create a function
that is able to identify faces in a given image. In case you haven't
noticed we're using images that are intended to solve the problem below.
The negative directory contains patches of a scene where there is people
in it except for their target face(s). We are using this data to bias
the classifier to find a face on a specific scene in order to reduce
computation time (you will need a larger amount of positive and negative
examples for a more robust classifier).
Use a 24x24 sliding window and check if it is identified as a face. If
this is the case, draw a 24 x 24 rectangle to highlight positive match.
You should be able to only find the man's face. To that extent you will
need to define a positive and negative datasets for the face detection.
You can choose images from the pos/ and neg/ datasets or build your own
from the man.jpeg image by selecting subimages.
**Code:** ViolaJones.faceDetection(image, filename=None)

**Report:**
Use the following input images and return a copy with the highlighted
face regions.
-Input: man.jpeg. Output:ps4-4-c-1.png\

# Extra Credit: Cascade Classifier \[10 points\]

In this last section, we will use the Haar-features and Viola Jones
classifier to build a cascade of classifiers for face detection. This
section is extra credit.

## 

First, we will write the functions predict and evaluates_classifiers to
evaluate a set of cascaded classifiers. We represent the set of cascaded
classifiers as a list where each index is a Viola Jones classifier
object. Within the cascade, only positive results from the i'th
classifier get passed to the i+1 classifier. At any stage, if a
classifier returns negative, then the entire cascade returns negative.
The predict function takes an image and returns positive or negative
based on if a face is present. The evaluates_classifiers takes a set of
positive and negatives images and reports the detection rate, false
positive rate, and a list of the false positive images.

**Code:**

-   CascadeClassifier.predict(classifiers, img)

-   CascadeClassifier.evaluate_classifiers(pos, neg, classifiers)

## 

Now we will write the training routine for the cascaded classifier. Use
the Cascade algorithm in the Viola-Jones paper (labeled as table 2 in
the paper) to build the cascaded classifier. We've provided an adapted
version for this problem set below. Also complete the face detection
function similar to ViolaJones.

**Code:**

-   CascadeClassifier.train(classifiers, img)

-   CascadeClassifier.faceDetection(image)

**Report:**
-Text Answer: Report the cascaded classifier accuracy on both the
training and test sets. What was the best percentage for the train/test
split? What values did you choose for the false positive target, the
false positive rate, and the detection rate? What impact did these have
on the overall cascaded classifier?\
-Text Answer: How many classifiers did your cascade algorithm produce?
How many features did each of these classifiers have? Compare this
classifier to just a single Viola Jones classifiers.
-Image Answer: Include an image you selected and the faces detected on
the image. Choose any image of your liking except for the image given in
section 4 (man.jpeg).
