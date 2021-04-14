import cv2 as cv
import numpy as np
from PIL import Image

two_stacked_frames = []

# The video feed is read in as
# a VideoCapture object
cap = cv.VideoCapture("data/TV-HI/highFive_0028.avi")

middle_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT)/ 2)

# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
ret, first_frame = cap.read()

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 255

counter = 1
stacked_frames = []
while (cap.isOpened()):
    # ret = a boolean return value from getting
    # the frame, frame = the current frame being
    # projected in the video
    ret, frame = cap.read()

    if frame is not None:
        # Opens a new window and displays the input
        # frame
        cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if counter >= middle_frame - 10 and counter <= middle_frame + 9:
            # Calculates dense optical flow by Farneback method
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            horz = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
            vert = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
            horz = horz.astype('uint8')
            vert = vert.astype('uint8')
            # cv.imshow('Horizontal Component', horz)
            # cv.imshow('Vertical Component', vert)
            stacked_frames.append(horz)

            # Computes the magnitude and angle of the 2D vectors
            # magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
            # mag_angle_array = np.ndarray(shape=(336,608), dtype=object)
            # for y in range(336):
            #     for x in range(608):
            #         mag_ang = np.array([magnitude[y][x], angle[y][x]])
            #         mag_angle_array[y][x] = mag_ang
            # stacked_frames.append(mag_angle_array)
            # np.save("test_array",mag_angle_array)

            # Sets image hue according to the optical flow
            # direction
            # mask[..., 0] = angle * 180 / np.pi / 2
            #
            # # Sets image value according to the optical flow
            # # magnitude (normalized)
            # mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
            #
            # # Converts HSV to RGB (BGR) color representation
            # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
            #
            # # Opens a new window and displays the output frame
            # cv.imshow("dense optical flow", rgb)

        # Updates previous frame
        prev_gray = gray

        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        counter = counter + 1
    else:
        cap.release()

# np.save("test_array_twenty_frames",array )
two_stacked_frames.append(np.asarray(stacked_frames))