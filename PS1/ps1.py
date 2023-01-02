import math
import numpy as np
import cv2

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    image = np.copy(image)

    image_red = image[:, :, 2].copy()
    return image_red


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    image = np.copy(image)

    image_green = image[:, :, 1].copy()
    return image_green


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    image = np.copy(image)

    image_blue = image[:, :, 0].copy()
    return image_blue


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    image = np.copy(image)

    image_green = image[:, :, 1].copy()
    image_blue = image[:, :, 0].copy()
    image[:, :, 0] = image_green
    image[:, :, 1] = image_blue
    return image


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    src = np.copy(src)
    dst = np.copy(dst)

    row_start = (len(src) - shape[0]) // 2
    col_start = (len(src[0]) - shape[1]) // 2
    size_100_x_100_copy = src[
        row_start : row_start + shape[0], 
        col_start : col_start + shape[1]
    ].copy()

    image = dst.copy()
    row_start = (len(dst) - shape[0]) // 2
    col_start = (len(dst[0]) - shape[1]) // 2
    image[row_start : row_start + shape[0], col_start : col_start + shape[1]] = size_100_x_100_copy
    return image



def copy_paste_middle_circle(src, dst, radius):
    """ Copies the middle circle region of radius "radius" from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    Args:
        src (numpy.array): 2D array where the circular shape will be copied from.
        dst (numpy.array): 2D array where the circular shape will be copied to.
        radius (scalar): scalar value of the radius.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    src = np.copy(src)
    dst = np.copy(dst)
    radius = int(radius)

    center_x = len(src[0]) / 2
    if float(center_x).is_integer():
        center_x -= 1
    else:
        center_x = math.ceil(center_x)

    center_y = len(src) // 2
    if float(center_y).is_integer():
        center_y -= 1
    else:
        center_y = math.ceil(center_y)
    mask = np.zeros_like(src)
    mask = cv2.UMat.get(cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1))
    # mask = cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    src = np.array(src, dtype=np.uint8)
    mask = np.array(mask, dtype=np.uint8)
    image_one = np.multiply(mask, src)

    image = dst.copy()

    center_x = len(image[0]) // 2
    if float(center_x).is_integer():
        center_x -= 1
    else:
        center_x = math.ceil(center_x)

    center_y = len(image) // 2
    if float(center_y).is_integer():
        center_y -= 1
    else:
        center_y = math.ceil(center_y)
        
    mask = np.zeros_like(image) + 1
    mask = cv2.UMat.get(cv2.circle(mask, (center_x, center_y), radius, (0, 0, 0), -1))
    # mask = cv2.circle(mask, (center_x, center_y), radius, (0, 0, 0), -1)
    image = np.array(image, dtype=np.uint8)
    mask = np.array(mask, dtype=np.uint8)
    image_two = np.multiply(mask, image)

    crop_factor_columns = (len(image_one[0]) - 2 * radius) // 2
    image_one = image_one[:, crop_factor_columns : crop_factor_columns + 2 * radius]
    crop_factor_rows = (len(image_one) - 2 * radius) // 2
    image_one = image_one[crop_factor_rows : crop_factor_rows + 2 * radius, :]

    crop_factor_columns = (len(image_two[0]) - 2 * radius) // 2
    crop_factor_rows = (len(image_two) - 2 * radius) // 2

    image_two[crop_factor_rows : crop_factor_rows + 2 * radius, crop_factor_columns : crop_factor_columns + 2 * radius] += image_one
    return image_two


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    image = np.copy(image)

    min = np.float64(np.min(image))
    max = np.float64(np.max(image))
    mean = np.mean(image)
    stddev = np.std(image)
    return (min, max, mean, stddev)


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    image = np.copy(image)
    mean = np.mean(image)
    stddev = np.std(image)

    image = image - mean
    stddev_new = scale / stddev
    image = mean + image * stddev_new
    return image


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    image = np.copy(image)
    image_concat = np.array(image[:, -1].copy(), dtype=np.uint8)
    image = np.delete(image, np.s_[0:shift:1], axis=1)
    image = np.array(image, dtype=np.uint8)
    h_concat_list = []
    h_concat_list.append(image)
    for i in range(shift):
        h_concat_list.append(image_concat)
    image = cv2.hconcat(h_concat_list)
    return image


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    img1 = np.copy(img1)
    img2 = np.copy(img2)

    image = img1 - img2
    min = np.min(image)
    image = image - min

    max = np.max(image)
    if max != 0:
        image = image * 255 / max
    return image


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    image = np.copy(image)
    rows, columns = image.shape[:2]

    gaussian_noise = np.random.normal(0, sigma, [rows, columns])
    image[:, :, channel] = np.add(image[:, :, channel], gaussian_noise, out=image[:, :, channel], casting="unsafe")
    return image


def build_hybrid_image(image1, image2, cutoff_frequency):
    """ 
    Takes two images and creates a hybrid image given a cutoff frequency.
    Args:
        image1: numpy nd-array of dim (m, n, c)
        image2: numpy nd-array of dim (m, n, c)
        cutoff_frequency: scalar
    
    Returns:
        hybrid_image: numpy nd-array of dim (m, n, c)

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """

    filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                                   sigma=cutoff_frequency)
    filter = np.dot(filter, filter.T)
    
    low_frequencies = cv2.filter2D(image1,-1,filter)

    high_frequencies = image2 - cv2.filter2D(image2,-1,filter)
    
    return high_frequencies + low_frequencies


def vis_hybrid_image(hybrid_image):
    """ 
    Tools to visualize the hybrid image at different scale.

    Credits:
        Assignment developed based on a similar project by James Hays. 
    """


    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales+1):
      # add padding
      output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                          dtype=np.float32)))

      # downsample image
      cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

      # pad the top to append to the output
      pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                     num_colors), dtype=np.float32)
      tmp = np.vstack((pad, cur_image))
      output = np.hstack((output, tmp))

    return output
