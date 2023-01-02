import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.
    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.
    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.
    It is recommended you use Hough tools to find these circles in
    the image.
    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.
    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.
    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    minimum = int(radii_range[0])
    maximum = int(radii_range[-1]) + 1

    img_in_greyscale = cv2.cvtColor(img_in.copy(), cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        img_in_greyscale, 
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1 = 4,
        param2 = 10,
        minRadius = minimum,
        maxRadius = maximum
    )

    circle_list = []
    for circle in circles[0]:
        x, y, radius = circle
        ## Crop square from center of circle
        circle_image = img_in[int(y - radius / 2) : int(y + radius / 2), int(x - radius / 2) : int(x + radius / 2)].copy()
        circle_dict = {'x': x, 'y': y, 'radius': radius, 'blue': circle_image[0, 0, 0], 'green': circle_image[0, 0, 1], 'red': circle_image[0, 0, 2]}
        circle_list.append(circle_dict)

    for circle in circle_list:
        if circle['green'] in [127, 255] and circle['red'] in [127, 255]:
            coordinates = (circle['x'], circle['y'])
            circle['color'] = 'yellow'
            if 255 in [circle['green'], circle['red'], circle['blue']]:
                state = 'yellow'
        elif circle['green'] in [127, 255]:
            circle['color'] = 'green'
            if 255 in [circle['green'], circle['red'], circle['blue']]:
                state = 'green'
        else:
            circle['color'] = 'red'
            if 255 in [circle['green'], circle['red'], circle['blue']]:
                state = 'red'

    return (coordinates, state)


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img_in_greyscale = cv2.cvtColor(img_in.copy(), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_in_greyscale, 200, 200, apertureSize = 3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        45
    )
    
    line_list = [line[0] for line in lines]
    line_list_x_values_summed = 0
    line_list_y_values_summed = 0
    
    for line in line_list:
        line_list_x_values_summed += line[0] + line[2]
        line_list_y_values_summed += line[1] + line[3]

    line_list_x_values_summed /= len(line_list) * 2
    line_list_y_values_summed /= len(line_list) * 2
    
    coordinates = (int(np.round(line_list_x_values_summed)), int(np.round(line_list_y_values_summed)))

    return coordinates


def template_match(img_orig, img_template, method):
    """Returns the location corresponding to match between original image and provided template.
    Args:
        img_orig (np.array) : numpy array representing 2-D image on which we need to find the template
        img_template: numpy array representing template image which needs to be matched within the original image
        method: corresponds to one of the four metrics used to measure similarity between template and image window
    Returns:
        Co-ordinates of the topmost and leftmost pixel in the result matrix with maximum match
    """
    """Each method is calls for a different metric to determine 
       the degree to which the template matches the original image
       We are required to implement each technique using the 
       sliding window approach.
       Suggestion : For loops in python are notoriously slow
       Can we find a vectorized solution to make it faster?
    """
    result = np.zeros(
        (
            (img_orig.shape[0] - img_template.shape[0] + 1),
            (img_orig.shape[1] - img_template.shape[1] + 1),
        ),
        float,
    )
    top_left = []
    
    img_orig = np.array(img_orig, dtype=np.float64)
    img_template = np.array(img_template, dtype=np.float64)
    height_template = img_template.shape[0]
    width_template = img_template.shape[1]

    slider_y = img_orig.shape[0] - img_template.shape[0] + 1
    slider_x = img_orig.shape[1] - img_template.shape[1] + 1

    """Once you have populated the result matrix with the similarity metric corresponding to each overlap, return the topmost and leftmost pixel of
    the matched window from the result matrix. You may look at Open CV and numpy post processing functions to extract location of maximum match"""
    # Sum of squared differences
    if method == "tm_ssd":
        for y in range(slider_y):
            for x in range(slider_x):
                value = np.sum((img_orig[y : y + height_template, x : x + width_template] - img_template)**2)
                result[y, x] = value

        result_flattened = result.flatten()

        min_value = np.argmin(result_flattened)
        col_of_min = int(math.floor(min_value / slider_x))
        row_of_min = int(min_value - col_of_min * slider_x)

        return (row_of_min, col_of_min)

    # Normalized sum of squared differences
    elif method == "tm_nssd":
        for y in range(slider_y):
            for x in range(slider_x):
                value = np.sum((img_orig[y : y + height_template, x : x + width_template] - img_template)**2)
                result[y, x] = value

        result -= np.min(result)
        result /= np.max(result)

        result_flattened = result.flatten()

        min_value = np.argmin(result_flattened)
        col_of_min = int(math.floor(min_value / slider_x))
        row_of_min = int(min_value - col_of_min * slider_x)

        return (row_of_min, col_of_min)

    # Cross Correlation
    elif method == "tm_ccor":
        img_orig_edges = cv2.Canny(np.uint8(img_orig), 50, 150, apertureSize = 3)
        img_template_edges = cv2.Canny(np.uint8(img_template), 50, 150, apertureSize = 3)
        for y in range(slider_y):
            for x in range(slider_x):
                value = np.sum(np.multiply(img_orig_edges[y : y + height_template, x : x + width_template], img_template_edges))
                result[y, x] = value

        result -= np.min(result)
        result /= np.max(result)

        result_flattened = result.flatten()

        max_value = np.argmax(result_flattened)
        col_of_max = int(math.floor(max_value / slider_x))
        row_of_max = int(max_value - col_of_max * slider_x)

        return (row_of_max, col_of_max)

    # Normalized Cross Correlation
    elif method == "tm_nccor":
        img_orig_edges = cv2.Canny(np.uint8(img_orig), 50, 150, apertureSize = 3)
        img_template_edges = cv2.Canny(np.uint8(img_template), 50, 150, apertureSize = 3)
        for y in range(slider_y):
            for x in range(slider_x):
                value = np.sum(np.multiply(img_orig_edges[y : y + height_template, x : x + width_template], img_template_edges))
                result[y, x] = value

        result -= np.min(result)
        result /= np.max(result)

        result_flattened = result.flatten()

        max_value = np.argmax(result_flattened)
        col_of_max = int(math.floor(max_value / slider_x))
        row_of_max = int(max_value - col_of_max * slider_x)

        # cv2.circle(result, (row_of_max, col_of_max), 5, (255, 255, 255), 10)
        # cv2.imshow('cross corr', result)
        # cv2.waitKey(0)

        return (row_of_max, col_of_max)

    else:
        """Your code goes here"""
        # Invalid technique
    return None
    ## return top_left


'''Below is the helper code to print images for the report'''
#     cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(result,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img_orig,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(method)
#     plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    x = np.asarray(x, dtype=np.complex_)
    len_of_sequence = len(x)
    sample = np.array([i for i in range(len_of_sequence)])
    curr_freq = np.array([sample]).T
    constant = np.exp(-2j * np.pi * curr_freq * sample / len_of_sequence)
    y = np.dot(constant, x)
    return y


def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    x = np.asarray(x, dtype=np.complex_)

    # len_of_sequence = len(x)
    # sample = np.array([i for i in range(len_of_sequence)])
    # y = []
    # for n in range(len_of_sequence):
    #     value = 0
    #     for k, i in zip(x, sample):
    #         value += 1 / len_of_sequence * np.exp(2j * np.pi * n * k / len_of_sequence)
    #     y.append(value)
    # y = np.array(y)

    len_of_sequence = len(x)
    sample = np.array([i for i in range(len_of_sequence)])
    curr_freq = np.array([sample]).T
    constant = np.exp(2j * np.pi * curr_freq * sample / len_of_sequence)
    y = 1 / len_of_sequence * np.dot(constant, x)    
    return y


def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,n) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,n) representing Fourier-Transformed image

    """
    img = np.array(img, dtype=np.complex_)

    ## Replace all columns using iDFT
    for i in range(len(img[0])):
        img[:, i] = np.array(dft(img[:, i]))

    ## Replace all rows using iDFT
    for j in range(len(img)):
        img[j, :] = np.array(dft(img[j, :]))

    return img

def idft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,n) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,n) representing image

    """
    img = np.array(img, dtype=np.complex_)

    ## Replace all columns using iDFT
    for i in range(len(img[0])):
        img[:, i] = np.array(idft(img[:, i]))

    ## Replace all rows using iDFT
    for j in range(len(img)):
        img[j, :] = np.array(idft(img[j, :]))

    return img

def get_thresholded_channel(img, threshold_percentage):
    shape = img.shape
    # threshold_percentage = (1 - threshold_percentage) * 100
    # color_channel = img.flatten()
    # threshold_value = np.percentile(abs(color_channel), threshold_percentage)
    # color_channel = [0 if abs(i) < threshold_value else i for i in color_channel]
    # color_channel = np.reshape(color_channel, shape)
    
    sorted_channel = np.sort(abs(img.flatten()))
    total_count = len(sorted_channel)
    cutoff_value_index = int((1 - threshold_percentage) * total_count)
    cutoff_value = sorted_channel[int(cutoff_value_index)]
    color_channel = (abs(img) > cutoff_value) * img
    color_channel = np.reshape(color_channel, shape)

    return color_channel

def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,n,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,n,3) representing compressed image
        compressed_frequency_img (np.array): numpy array of shape (n,n,3) representing the compressed image in the frequency domain

    """
    img_bgr = np.array(img_bgr, dtype=np.complex_)

    for i in range(3):
        color_channel = img_bgr[:, :, i].copy()
        color_channel = np.fft.fft2(color_channel)
        img_bgr[:, :, i] = np.fft.fftshift(color_channel)
        img_bgr[:, :, i] = get_thresholded_channel(img_bgr[:, :, i], threshold_percentage)

    compressed_frequency_img = img_bgr.copy()

    cv2.imshow('freq domain', np.abs(compressed_frequency_img))
    cv2.waitKey(0)

    for i in range(3):
        color_channel = np.fft.ifftshift(img_bgr[:, :, i].copy())
        img_bgr[:, :, i] = np.fft.ifft2(color_channel)

    img_compressed = abs(img_bgr.copy())

    cv2.imshow('spatial domain', np.abs(img_compressed))
    cv2.waitKey(0)

    return img_compressed, compressed_frequency_img


def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,n,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,n,3) representing low pass filtered image
        low_pass_frequency_img (np.array): numpy array of shape (n,n,3) representing the low pass filtered image in the frequency domain

    """
    img_bgr = np.array(img_bgr, dtype=np.complex_)

    for i in range(3):
        img_bgr[:, :, i] = np.fft.fftshift(np.fft.fft2(img_bgr[:, :, i].copy()))

    img_mask = np.zeros((img_bgr.shape[0], img_bgr.shape[1]))

    img_mask = np.array(img_mask, dtype=np.float64)
    center = (
        int(np.floor(len(img_mask[0]) / 2)), ## col
        int(np.floor(len(img_mask) / 2)) ## row
    )
    cv2.circle(img_mask, center, r, (1, 1, 1), -1)

    img_mask = np.dstack((img_mask, img_mask, img_mask))

    img_bgr = img_bgr * img_mask

    low_pass_frequency_img = img_bgr.copy()

    for i in range(3):
        color_channel = np.fft.ifftshift(img_bgr[:, :, i].copy())
        img_bgr[:, :, i] = np.fft.ifft2(color_channel)

    img_low_pass = np.abs(img_bgr.copy())

    return img_low_pass, low_pass_frequency_img
