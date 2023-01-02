"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

import cv2
import numpy as np
import math
#from typing import Tuple

from matplotlib import pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 2)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        print('Saved')

        np.save(fr'{ROOT_DIR}/p1.npy', points_1)
        np.save(fr'{ROOT_DIR}/p2.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """
    euc_dis = np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
    return euc_dis


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    height, width = image.shape[:2]
    corners = [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]
    return corners

def get_row_col_avg(quadrant):
    rows, cols = np.where(quadrant[1] == 255)
    rows = np.array(rows)
    cols = np.array(cols)

    if quadrant[0] == 1:
        avg_row = int(np.mean(rows))
        avg_col = int(np.mean(cols))
    
    elif quadrant[0] == 2:
        avg_row = int(np.mean(rows) + len(quadrant[1]))
        avg_col = int(np.mean(cols))

    elif quadrant[0] == 3:
        avg_row = int(np.mean(rows))
        avg_col = int(np.mean(cols) + len(quadrant[1][0]))

    else:
        avg_row = int(np.mean(rows) + len(quadrant[1]))
        avg_col = int(np.mean(cols) + len(quadrant[1][0]))  

    point = (avg_col, avg_row)
    return point

def get_corners(image):
    img_in_greyscale = cv2.cvtColor(np.uint8(image.copy()), cv2.COLOR_BGR2GRAY)
    img_in_greyscale = cv2.medianBlur(img_in_greyscale, 5)
    corners = cv2.cornerHarris(
        img_in_greyscale,
        5, ## keep 5 in gradescope; 10 locally 
        5,
        0.0001
    )
    corners = np.array(corners)
    ## 0.003 in Gradescope (maybe)
    ## 0.03 locally
    corners = (corners > 0.003) * np.ones((corners.shape)) * 255

    cv2.imshow('corners', corners)
    cv2.waitKey(0)
    
    num_of_cols = corners.shape[0] // 2
    num_of_rows = corners.shape[1] // 2
    quadrants = [
        corners[col : col + num_of_cols, row : row + num_of_rows]
        for col in range(0, corners.shape[0], num_of_cols)
        for row in range(0, corners.shape[1], num_of_rows)
    ]
    
    top_left_quadrant = quadrants[0]
    bottom_left_quadrant = quadrants[2]
    top_right_quadrant = quadrants[1]
    bottom_right_quadrant = quadrants[3]

    points = [(0, 0)] * 4
    reordered_quadrants = [[1, top_left_quadrant], [2, bottom_left_quadrant], [3, top_right_quadrant], [4, bottom_right_quadrant]]
    for i in range(4):
        points[i] = get_row_col_avg(reordered_quadrants[i])

    return points

def find_markers(image, template):
    """Finds four corner markers.

    Use a combination of circle finding and/or corner finding and/or convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    corners = get_corners(image)
    return corners


def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """

    image = cv2.line(image, markers[0], markers[2], (255, 255, 255), thickness)
    image = cv2.line(image, markers[2], markers[3], (255, 255, 255), thickness)
    image = cv2.line(image, markers[3], markers[1], (255, 255, 255), thickness)
    image = cv2.line(image, markers[1], markers[0], (255, 255, 255), thickness)

    return image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """
    in_image = imageA.copy()
    out_image = imageB.copy()

    for y in range(len(out_image)):
        for x in range(len(out_image[0])):
            b = np.array([[x, y, 1]]).T
            a = np.dot(np.linalg.pinv(homography), b)
            new_x = int(a[0] / a[2])
            new_y = int(a[1] / a[2])
            if new_x < 0 or new_y < 0 or new_x >= len(in_image[0]) or new_y >= len(in_image):
                continue
            else:
                out_image[y, x] = in_image[new_y, new_x]

    return out_image


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """
    m1 = np.zeros((8, 9), dtype=np.float64)

    for i in range(len(srcPoints)):
        xa = srcPoints[i][0]
        ya = srcPoints[i][1]
        xb = dstPoints[i][0]
        yb = dstPoints[i][1]
        m1[i * 2 : i * 2 + 2, :] = np.array([
            [-xa, -ya, -1, 0, 0, 0, xa * xb, ya * xb, xb],
            [0, 0, 0, -xa, -ya, -1, xa * yb, ya * yb, yb]
        ])

    homography = np.linalg.svd(m1)[2]
    homography = homography[-1].reshape(3, 3)

    return homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """
    video = cv2.VideoCapture(filename)
    while(True):
        exists, frame = video.read()
        if exists == True:
            yield frame
        else:
            video.release()
            yield None



class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)


    def filter(self, img, filter, padding=(0,0)):
        raise NotImplementedError
        return output

    def gradients(self, image_bw):
        border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
        Ix = cv2.Sobel(np.uint8(image_bw.copy()), cv2.CV_64F, dx = 1, dy = 0, borderType = border)
        Iy = cv2.Sobel(np.uint8(image_bw.copy()), cv2.CV_64F, dx = 0, dy = 1, borderType = border)

        return Ix, Iy

    def get_gaussian(self, ksize, sigma):
        kernel = cv2.getGaussianKernel(ksize, sigma)
        return kernel

    
    def second_moments(self, image_bw, ksize=7, sigma=10):
        Ix, Iy = self.gradients(image_bw)
        sx2, sy2, sxsy = Ix * Ix, Iy * Iy, Ix * Iy
        kernel = self.get_gaussian(ksize, sigma)
        print(kernel)
        return sx2, sy2, sxsy

    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        R = None
        # raise NotImplementedError
        return R

    def pool2d(self, A, kernel_size, stride, padding, pool_mode='max'):
        '''
        2D Pooling
        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        '''
        # Padding
        A = np.pad(A, padding, mode='constant')

        # Window view of A
        output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                        (A.shape[1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A_w = np.lib.stride_tricks.as_strided(A, shape = output_shape + kernel_size, 
                            strides = (stride*A.strides[0],
                                    stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        if pool_mode == 'max':
            return A_w.max(axis=(1,2)).reshape(output_shape)
        elif pool_mode == 'avg':
            return A_w.mean(axis=(1,2)).reshape(output_shape)
            

    def nms_maxpool_numpy(self, R: np.ndarray, k, ksize):
        """Pooling function that takes in an array input
        Args:
            R (np.ndarray): Harris Response Map
            k (int): the number of corners that are to be detected with highest probability
            ksize (int): pooling size
        Return:
            x: x indices of the corners
            y: y indices of the corners
        """
        x, y = None, None
        # raise NotImplementedError
        return x, y

    def harris_corner(self, image_bw, k=100):
        """Harris Corner Detection Function that takes in an image and detects the most likely k corner points.
        Args:
            image_bw (np.array): black and white image
            k (int): top k number of corners with highest probability to be detected by Harris Corner
        RReturn:
            x: x indices of the top k corners
            y: y indices of the top k corners
        """   
        x, y = None, None
        # raise NotImplementedError
        return x, y


    def calculate_num_ransac_iterations(
            self,prob_success: float, sample_size: int, ind_prob_correct: float):

        num_samples = None

        p = prob_success
        s = sample_size
        e = 1 - ind_prob_correct

        num_samples = np.log(1 - p) / np.log(1 - (1 - e) ** s)
        print('Num of iterations', int(num_samples))

        return int(round(num_samples))


    def ransac_homography_matrix(self, matches_a: np.ndarray, matches_b: np.ndarray):

        p = 0.999
        s = 8
        sample_size_iter = 8
        e = 0.5
        threshold = 1
        numi = self.calculate_num_ransac_iterations(p, s, e)

        org_matches_a = matches_a
        org_matches_b = matches_b
        print('matches', org_matches_a.shape, org_matches_b.shape)
        matches_a = np.hstack([matches_a, np.ones([matches_a.shape[0], 1])])
        matches_b = np.hstack([matches_b, np.ones([matches_b.shape[0], 1])])
        in_list = []
        in_sum = 0
        best_in_sum = -99
        inliers = []
        final_inliers = []

        y = Image_Mosaic().get_homography_parameters(org_matches_b, org_matches_a)

        best_F = np.full_like(y, 1)
        choice = np.random.choice(org_matches_a.shape[0], sample_size_iter)
        print('s',org_matches_b[choice].shape,matches_b[choice].shape)
        best_inliers = np.dot(matches_a[choice], best_F) - matches_b[choice]
        print('inliers shape',best_inliers.shape,best_inliers)

        count = 0
        for i in range(min(numi, 20000)):
            
            choice = np.random.choice(org_matches_a.shape[0], sample_size_iter)
            match1, match2 = matches_a[choice], matches_b[choice]


            F = Image_Mosaic().get_homography_parameters(match2, match1)

            count += 1
            inliers = np.dot(matches_a[choice], F)- matches_b[choice]

            inliers = inliers[np.where(abs(inliers) <= threshold)]

            in_sum = abs(inliers.sum())
            best_in_sum = max(in_sum, best_in_sum)
            best_inliers = best_inliers if in_sum < best_in_sum else inliers

            if abs(in_sum) >= best_in_sum:
                # helper to debug
                # print('insum', in_sum)
                pass

            best_F = best_F if abs(in_sum) < abs(best_in_sum) else F


        for j in range(matches_a.shape[0]):
            final_liers = np.dot(matches_a[j], best_F) - matches_b[j]
            final_inliers.append(abs(final_liers) < threshold)

        final_inliers = np.stack(final_inliers)

        inliers_a = org_matches_a[np.where(final_inliers[:,0]==True) and np.where(final_inliers[:,1]==True)]
        inliers_b = org_matches_b[np.where(final_inliers[:,0]==True) and np.where(final_inliers[:,1]==True)]

        print('best F', best_F.shape, inliers_a.shape, inliers_b.shape, best_F, inliers_a, inliers_b)

        return best_F, inliers_a, inliers_b


class Image_Mosaic(object):

    def __int__(self):
        pass
    
    def image_warp_inv(self, im_src, im_dst, homography):
        raise NotImplementedError

    def output_mosaic(self, img_src, img_warped):
        
        raise NotImplementedError
        return im_mos_out

    def get_homography_parameters(self, points2, points1):
        """
        leverage your previous implementation of 
        find_four_point_transform() for this part.
        """
        num_of_points = len(points2)
        m1 = np.zeros((num_of_points * 2, 9), dtype=np.float64)

        for i in range(num_of_points):
            xa = points2[i][0]
            ya = points2[i][1]
            xb = points1[i][0]
            yb = points1[i][1]
            m1[i * 2 : i * 2 + 2, :] = np.array([
                [-xa, -ya, -1, 0, 0, 0, xa * xb, ya * xb, xb],
                [0, 0, 0, -xa, -ya, -1, xa * yb, ya * yb, yb]
            ])

        homography = np.linalg.svd(m1)[2]
        homography = homography[-1].reshape(3, 3)

        return homography






