"""Problem Set 4: Motion Detection"""

import cv2
import numpy as np
from scipy import signal


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    ksize = 3
    scale = 1 / 8
    X_gradient = cv2.Sobel(
        np.float32(image),
        ddepth = cv2.CV_32F,
        dx = 1,
        dy = 0,
        ksize = ksize,
        scale = scale
    )
    return X_gradient


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    ksize = 3
    scale = 1 / 8
    Y_gradient = cv2.Sobel(
        np.float32(image),
        ddepth = cv2.CV_32F,
        dx = 0,
        dy = 1,
        ksize = ksize,
        scale = scale
    )
    return Y_gradient


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    gaussian_kernel = cv2.getGaussianKernel(k_size, sigma)
    uniform_kernel = np.ones((k_size, k_size)) 

    image_one = img_a.copy()
    image_two = img_b.copy()

    fx = gradient_x(image_one)
    fy = gradient_y(image_one)

    if k_type == 'gaussian':
        image_one = signal.convolve2d(image_one, gaussian_kernel, mode = 'same')
        image_two = signal.convolve2d(image_two, gaussian_kernel, mode = 'same')
    else:
        image_one = signal.convolve2d(image_one, uniform_kernel, mode = 'same') * 15 / (k_size ** 2) 
        image_two = signal.convolve2d(image_two, uniform_kernel, mode = 'same') * 15 / (k_size ** 2)

    ft = image_two - image_one
    
    U = np.zeros((image_one.shape[0], image_one.shape[1], 2, 1))

    top_left_A = fx * fx
    bottom_left_A = fx * fy
    top_right_A = fx * fy
    bottom_right_A = fy * fy
    
    top_b = -fx * ft
    bottom_b = -fy * ft

    window_size = np.floor(k_size / 2)
    window_size = int(window_size)

    for i in range(window_size, image_one.shape[1] - window_size - 1):
        for j in range(window_size, image_one.shape[0] - window_size - 1):
            top_left_A_sum = np.sum(top_left_A[j - window_size : j + window_size + 1, i - window_size : i + window_size + 1])
            bottom_left_A_sum = np.sum(bottom_left_A[j - window_size : j + window_size + 1, i - window_size : i + window_size + 1])
            top_right_A_sum = np.sum(top_right_A[j - window_size : j + window_size + 1, i - window_size : i + window_size + 1])
            bottom_right_A_sum = np.sum(bottom_right_A[j - window_size : j + window_size + 1, i - window_size : i + window_size + 1])

            top_b_sum = np.sum(top_b[j - window_size : j + window_size + 1, i - window_size : i + window_size + 1])
            bottom_b_sum = np.sum(bottom_b[j - window_size : j + window_size + 1, i - window_size : i + window_size + 1])

            A_for_point = np.array([
                [top_left_A_sum, top_right_A_sum],
                [bottom_left_A_sum, bottom_right_A_sum]
            ])
            b_for_point = np.array([top_b_sum, bottom_b_sum]).T
    
            inverse = np.linalg.pinv(np.dot(A_for_point.T, A_for_point))
            inner_term = np.dot(inverse, A_for_point.T)
            u_for_point = np.dot(inner_term, b_for_point)
            U[j, i, :] = np.array([u_for_point]).T

    return U[:, :, 0], U[:, :, 1]


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    img = image.copy()
    new_img = np.zeros((int(np.ceil(img.shape[0] / 2)), int(np.ceil(img.shape[1] / 2))))
    
    kernel = np.array([1, 4, 6, 4, 1]) / 16
    img = cv2.sepFilter2D(src = img, ddepth = cv2.CV_64F, kernelX = kernel, kernelY = kernel, borderType = cv2.BORDER_DEFAULT)
    
    rows_to_keep = np.arange(0, len(img), 2)
    cols_to_keep = np.arange(0, len(img[0]), 2)
    
    img = img[rows_to_keep]
    for i in range(len(img)):
        new_img[i] = img[i][cols_to_keep]

    return new_img


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    img = image.copy()
    img_list = []
    img_list.append(img)
    for i in range(1, levels):
        img = reduce_image(img)
        img_list.append(img)

    return img_list


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    num_of_cols = np.sum(np.array([img.shape[1] for img in img_list]))
    num_of_rows = img_list[0].shape[0]
    out_img = np.zeros((num_of_rows, num_of_cols))
    
    col = 0
    for img in img_list:
        out_img[0 : img.shape[0], col : col + img.shape[1]] = img
        col += img.shape[1]
    
    return out_img


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    img = image.copy()

    new_img_equal_rows = np.zeros((
        img.shape[0],
        img.shape[1] * 2
    ))

    new_img = np.zeros((
        img.shape[0] * 2,
        img.shape[1] * 2
    ))

    kernel = np.array([1, 4, 6, 4, 1]) / 8

    for i in range(img.shape[0]):
        row = img[i]
        new_img_equal_rows[i, ::2] = row

    for i in range(new_img.shape[1]):
        col = new_img_equal_rows[:, i]
        new_img[::2, i] = col

    new_img = cv2.sepFilter2D(src = new_img, ddepth = cv2.CV_64F, kernelX = kernel, kernelY = kernel, borderType = cv2.BORDER_DEFAULT)

    return new_img


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    img_list = [None] * len(g_pyr)
    for i in range(1, len(g_pyr) + 1):
        idx = -i
        if idx == -1:
            img_list[idx] = g_pyr[idx]
        else:
            if g_pyr[idx].shape[0] % 2 == 1:
                img_list[idx] = g_pyr[idx] - expand_image(g_pyr[idx + 1])[:-1, :-1]
            else:
                img_list[idx] = g_pyr[idx] - expand_image(g_pyr[idx + 1])

    return img_list


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    img = image.copy()
    M, N = img.shape
    X, Y = np.meshgrid(range(N), range(M))

    X = np.array(X, dtype = np.uint8) + U
    Y = np.array(Y, dtype = np.uint8) + V

    img =  np.array(img, dtype = np.uint8)
    img = cv2.remap(img, X, Y, interpolation, borderMode = cv2.BORDER_REFLECT101)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    return img


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    raise NotImplementedError

def classify_video(images):
    """Classifies a set of frames as either
        - int(1) == "Running"
        - int(2) == "Walking"
        - int(3) == "Clapping"
    Args:
        images list(numpy.array): greyscale floating-point frames of a video
    Returns:
        int:  Class of video
    """
    raise NotImplementedError
