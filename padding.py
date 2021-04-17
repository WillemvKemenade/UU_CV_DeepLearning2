import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def add_padding(image, goal_height, goal_width, gray=False):

    height, width = get_size(image)

    # Empty goal image
    if gray:
        result = np.full((goal_height, goal_width), 0, dtype=np.uint8)
    else:
        result = np.full((goal_height, goal_width, 3), (0, 0, 0), dtype=np.uint8)

    # fit image into center of goal image
    x = (goal_width - width) // 2
    y = (goal_height - height) // 2
    result[y : y + height, x : x + width] = image

    # cv2.resize(img, (XSIZE, YSIZE), interpolation=cv2.INTER_NEAREST)

    # plt.imshow(image)
    # plt.show()
    # plt.imshow(result)
    # plt.show()
    
    return result

def pad_and_resize(image, resized_height, resized_width, gray=False):

    height, width = get_size(image)

    goal_height = 0
    goal_width = 0

    if height > width:
        goal_height = height
        goal_width = height
    else:
        goal_height = width
        goal_width = width

    # Empty goal image
    if gray:
        result = np.full((goal_height, goal_width), 0, dtype=np.uint8)
    else:
        result = np.full((goal_height, goal_width, 3), (0, 0, 0), dtype=np.uint8)

    # fit image into center of goal image
    x = (goal_width - width) // 2
    y = (goal_height - height) // 2
    result[y : y + height, x : x + width] = image

    resized_result = cv.resize(result, (resized_height, resized_width), interpolation=cv.INTER_NEAREST)

    # plt.imshow(image)
    # plt.show()
    # plt.imshow(resized_result)
    # plt.show()

    return resized_result

def get_max_size(images, max_height=0, max_width=0):
    for i in images:
        height, width = get_size(i)
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width
    print("max size =", max_height, max_width)
    return (max_height, max_width)

def get_size(image):
    return (image.shape[0], image.shape[1])