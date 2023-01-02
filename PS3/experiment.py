
"""
CS6476: Problem Set 3 Experiment file
This script consists  a series of function calls that run the ps3
implementation and output images so you can verify your results.
"""

import os
import numpy as np
import cv2
import copy
import PIL
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from ps3 import *

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

IMG_DIR = fr'{ROOT_DIR}/input_images'
VID_DIR = fr'{ROOT_DIR}/input_videos'
OUT_DIR = fr'{ROOT_DIR}/output'
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


def helper_for_part_4_and_5(video_name, fps, frame_ids, output_prefix,
                            counter_init, is_part5):

    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.__next__()
    h, w, d = image.shape

    out_path = "ar_{}-{}".format(output_prefix[4:], video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    if is_part5:
        advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
        src_points = get_corners_list(advert)

    output_counter = counter_init

    frame_num = 1

    while image is not None:

        print("Processing frame {}".format(frame_num))

        markers = find_markers(image, template)

        if is_part5:
            homography = find_four_point_transform(src_points, markers)
            image = project_imageA_onto_imageB(advert, image, homography)

        else:
            
            for marker in markers:
                mark_location(image, marker)

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1

        video_out.write(image)

        image = image_gen.__next__()

        frame_num += 1

    video_out.release()



def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)


def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)


def part_1():

    print("\nPart 1:")

    input_images = ['sim_clear_scene.jpg', 'sim_noisy_scene_1.jpg',
                    'sim_noisy_scene_2.jpg']
    output_images = ['ps3-1-a-1.png', 'ps3-1-a-2.png', 'ps3-1-a-3.png']

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        marker_positions = find_markers(scene, template)

        for marker in marker_positions:
            mark_location(scene, marker)

        save_image(img_out, scene)


def part_2():

    print("\nPart 2:")

    input_images = ['ps3-2-a_base.jpg', 'ps3-2-b_base.jpg',
                    'ps3-2-c_base.jpg', 'ps3-2-d_base.jpg', 'ps3-2-e_base.jpg']
    output_images = ['ps3-2-a-1.png', 'ps3-2-a-2.png', 'ps3-2-a-3.png',
                     'ps3-2-a-4.png', 'ps3-2-a-5.png']

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        markers = find_markers(scene, template)
        image_with_box = draw_box(scene, markers, 3)

        save_image(img_out, image_with_box)


def part_3():

    print("\nPart 3:")

    input_images = ['ps3-3-a_base.jpg', 'ps3-3-b_base.jpg', 'ps3-3-c_base.jpg']
    output_images = ['ps3-3-a-1.png', 'ps3-3-a-2.png', 'ps3-3-a-3.png']

    # Advertisement image
    advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
    src_points = get_corners_list(advert)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):
        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        markers = find_markers(scene, template)

        homography = find_four_point_transform(src_points, markers)

        projected_img = project_imageA_onto_imageB(advert, scene, homography)

        save_image(img_out, projected_img)


def part_4_a():

    print("\nPart 4a:")

    video_file = "ps3-4-a.mp4"
    frame_ids = [355, 555, 725]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 1, False)

    video_file = "ps3-4-b.mp4"
    frame_ids = [97, 407, 435]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-a", 4, False)


def part_4_b():

    print("\nPart 4b:")

    video_file = "ps3-4-c.mp4"
    frame_ids = [47, 470, 691]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-b", 1, False)

    video_file = "ps3-4-d.mp4"
    frame_ids = [207, 367, 737]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-4-b", 4, False)


def part_5_a():

    print("\nPart 5a:")

    video_file = "ps3-4-a.mp4"
    frame_ids = [355, 555, 725]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-a", 1, True)

    video_file = "ps3-4-b.mp4"
    frame_ids = [97, 407, 435]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-a", 4, True)


def part_5_b():

    print("\nPart 5b:")

    video_file = "ps3-4-c.mp4"
    frame_ids = [47, 470, 691]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-b", 1, True)

    video_file = "ps3-4-d.mp4"
    frame_ids = [207, 367, 737]
    fps = 40

    helper_for_part_4_and_5(video_file, fps, frame_ids, "ps3-5-b", 4, True)


    

def part_6(path1, path2):
    Mouse_Click_Correspondence().driver(path1, path2)


def part_7():
    ###load previously obtained points
    sxx1, sxx2, syy1, syy2 = [], [], [], []
    c1 = np.load(fr'{ROOT_DIR}/p1.npy')
    c2 = np.load(fr'{ROOT_DIR}/p2.npy')

    for cc1, cc2 in zip(c1, c2):
        sxx1.append(cc1[0])
        sxx2.append(cc2[0])
        syy1.append(cc1[1])
        syy2.append(cc2[1])

    sx1, sy1 = sxx1, syy1
    sx2, sy2 = sxx2, syy2

    points1, points2 = [], []
    for x, y in zip(sx1, sy1):
        points1.append((x, y))

    for x, y in zip(sx2, sy2):
        points2.append((x, y))

    homography_parameters = Image_Mosaic().get_homography_parameters(points2, points1)

    return homography_parameters


def part_8(path1, path2):
    def PIL_resize(img, size):

        img = numpy_arr_to_PIL_image(img, scale_to_255=True)
        img = img.resize(size)
        img = PIL_image_to_numpy_arr(img)
        return img

    def PIL_image_to_numpy_arr(img, downscale_by_255=True):

        img = np.asarray(img)
        img = img.astype(np.float32)
        if downscale_by_255:
            img /= 255
        return img

    def load_image(path: str) -> np.ndarray:
        img = PIL.Image.open(path)
        img = np.asarray(img, dtype=float)
        float_img_rgb = im2single(img)
        return float_img_rgb

    def im2single(im):

        im = im.astype(np.float32) / 255
        return im

    def rgb2gray(img):

        # Grayscale coefficients
        c = [0.299, 0.587, 0.114]
        return img[:, :, 0] * c[0] + img[:, :, 1] * c[1] + img[:, :, 2] * c[2]

    def numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: False):

        if scale_to_255:
            img *= 255

        return PIL.Image.fromarray(np.uint8(img))

    def save_image(path: str, im: np.ndarray) -> None:
        folder_path = os.path.split(path)[0]
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        img = copy.deepcopy(im)
        img = single2im(img)
        pil_img = numpy_arr_to_PIL_image(img, scale_to_255=False)
        pil_img.save(path)

    def show_interest_points(img, X, Y):

        newImg = img.copy()
        newImg *= 255
        newImg = PIL.Image.fromarray(np.uint8(newImg))
        r = 10
        draw = PIL.ImageDraw.Draw(newImg)
        for x, y in zip(X.astype(int), Y.astype(int)):
            cur_color = np.random.rand(3) * 255
            cur_color = (int(cur_color[0]), int(cur_color[1]), int(cur_color[2]))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=cur_color)

        pilimg = np.asarray(newImg)
        pilimg = pilimg.astype(np.float32)
        pilimg /= 255

        return pilimg

    def get_ransac_H(points_2d_pic_a, points_2d_pic_b):

        F, matched_points_a, matched_points_b = Automatic_Corner_Detection().ransac_homography_matrix(points_2d_pic_a,
                                                                                                       points_2d_pic_b)

        return F, matched_points_a, matched_points_b

    image1 = load_image(path1)
    image2 = load_image(path2)

    scale_factor = 1
    image1 = PIL_resize(image1, (int(image1.shape[1] * scale_factor), int(image1.shape[0] * scale_factor)))

    image2 = PIL_resize(image2, (int(image2.shape[1] * scale_factor), int(image2.shape[0] * scale_factor)))

    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    num_interest_points = 500 #You can experiment with this value
    X1, Y1 = Automatic_Corner_Detection().harris_corner(copy.deepcopy(image1_bw), num_interest_points)
    X2, Y2 = Automatic_Corner_Detection().harris_corner(copy.deepcopy(image2_bw), num_interest_points)

    points_2d_pic_a = []
    points_2d_pic_b = []
    for x, y in zip(X1, Y1):
        points_2d_pic_a.append((x, y))

    for x, y in zip(X2, Y2):
        points_2d_pic_b.append((x, y))

    points_2d_pic_a = np.stack(points_2d_pic_a)
    points_2d_pic_b = np.stack(points_2d_pic_b)

    homography_parameters_ransac, ptsa, ptsb = get_ransac_H(points_2d_pic_a, points_2d_pic_b)

    num_pts_to_visualize = 50
    rendered_img1 = show_interest_points(image1, ptsa[:num_pts_to_visualize, 0], ptsa[:num_pts_to_visualize, 1])
    rendered_img2 = show_interest_points(image2, ptsb[:num_pts_to_visualize, 0], ptsb[:num_pts_to_visualize, 1])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1);
    plt.imshow(rendered_img1)
    plt.subplot(1, 2, 2);
    plt.imshow(rendered_img2)
    print(f'{len(X1)} corners in image 1, {len(X2)} corners in image 2')
    plt.show()

    return homography_parameters_ransac


def part_9_a(homography_parameters, path1, path2):

    im_src = cv2.imread(path1, 1)
    im_dst = cv2.imread(path2, 1)

    cv2.namedWindow("Source Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Destination Image", cv2.WINDOW_NORMAL)

    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    
    
    im_warped = Image_Mosaic().image_warp_inv(im_src, im_dst, homography_parameters)


    cv2.namedWindow("Warped Source Image", cv2.WINDOW_NORMAL)
    cv2.imwrite(IMG_DIR + '\warped_image_a.jpg', im_warped)
    im_warped = cv2.normalize(im_warped, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow("Warped Source Image", im_warped)

    im_out_mosaic = Image_Mosaic().output_mosaic(im_src, im_warped)

    cv2.namedWindow("Output Mosaic Image", cv2.WINDOW_NORMAL)
    cv2.imwrite(IMG_DIR + '\image_mosiac_a.jpg', im_out_mosaic)
    cv2.imshow("Output Mosaic Image", im_out_mosaic)

    cv2.waitKey(0)


def part_9_b(homography_parameters, path1, path2):

    im_src = cv2.imread(path1, 1)
    im_dst = cv2.imread(path2, 1)

    cv2.namedWindow("Source Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Destination Image", cv2.WINDOW_NORMAL)

    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)

    ### Inverse Warping
    im_warped = Image_Mosaic().image_warp_inv(im_src,im_dst, homography_parameters)
                                           
    cv2.namedWindow("Warped Source Image", cv2.WINDOW_NORMAL)
    cv2.imwrite(IMG_DIR + '\warped_image_b.jpg', im_warped)
    im_warped = cv2.normalize(im_warped, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow("Warped Source Image", im_warped)

    im_out_mosaic = Image_Mosaic().output_mosaic(im_src, im_warped)
    cv2.namedWindow("Output Mosaic Image", cv2.WINDOW_NORMAL)
    cv2.imwrite(IMG_DIR + '\image_mosiac_b.jpg', im_out_mosaic)
    cv2.imshow("Output Mosaic Image", im_out_mosaic)

    cv2.waitKey(0)


if __name__ == '__main__':
    print("--- Problem Set 3 ---")
    # Comment out the sections you want to skip

    # part_1()
    part_2()
    # part_3()
    # part_4_a()
    # part_4_b()
    # part_5_a()
    # part_5_b()
    # part_5_c()
    
    path1 = fr'{IMG_DIR}/everest1.jpg'
    path2 = fr'{IMG_DIR}/everest2.jpg'

    # Part 6
    # part_6(path1, path2)
    
    # #Part 7
    # homography_parameters = part_7()
    # #Part 8
    # homography_parameters_ransac = part_8(path1, path2)
    # #Part 9
    # part_9_a(homography_parameters,path1,path2)
    # part_9_b(homography_parameters_ransac, path1, path2)




