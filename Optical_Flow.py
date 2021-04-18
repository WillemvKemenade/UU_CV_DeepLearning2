import cv2 as cv
import numpy as np

import padding

XSIZE = 64
YSIZE = 64
MAX_HEIGHT = 965
MAX_WIDTH = 997

def get_video_flow_stacks(video_list):
    stacked_videos = []
    index = 1
    print(str(len(video_list)) + "amount of videos")
    for val in video_list:
        print(str(index) + " out of " + str(len(video_list)))
        index = index + 1
        cap = cv.VideoCapture("data/TV-HI/"+val) #get the feed of the video_list
        middle_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT) / 2)
        ret, first_frame = cap.read()
        prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) # set the first frame to gray scale and use it for the first optical flow comparison
        # prev_gray = padding.add_padding(prev_gray, MAX_HEIGHT, MAX_WIDTH, gray=True)
        prev_gray = padding.pad_and_resize(prev_gray, YSIZE, XSIZE, gray=True)
        # prev_gray = cv.resize(prev_gray, (XSIZE, YSIZE), interpolation=cv.INTER_NEAREST)
        counter = 1
        stacked_frames = []

        while (cap.isOpened()):
            ret, frame = cap.read()
            if frame is not None:
                # cv.imshow("input", frame)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #gray scale of the next frame
                # gray = cv.resize(gray, (XSIZE, YSIZE), interpolation=cv.INTER_NEAREST)
                # gray = padding.add_padding(gray, MAX_HEIGHT, MAX_WIDTH, gray=True)
                gray = padding.pad_and_resize(gray, YSIZE, XSIZE, gray=True)

                if counter >= middle_frame - 10 and counter <= middle_frame + 9:
                    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    horz = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
                    vert = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
                    horz = horz.astype('uint8')
                    vert = vert.astype('uint8')
                    # cv.imshow('Horizontal Component', horz)
                    # cv.imshow('Vertical Component', vert)
                    stacked_frames.append(horz/255)
                    stacked_frames.append(vert/255)

                # Updates previous frame
                prev_gray = gray
                counter = counter + 1
            else:
                cap.release()

        frame_stack = np.asarray(stacked_frames)
        frame_stack = np.transpose(frame_stack, (1,2,0))
        stacked_videos.append(frame_stack)

    return stacked_videos

def get_middle_frames(video_list):
    image_list = []
    index = 1
    print(str(len(video_list)) + "amount of videos")
    for val in video_list:
        print(str(index) + " out of " + str(len(video_list)))
        index = index + 1

        cap = cv.VideoCapture("data/TV-HI/" + val)
        middle_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT) / 2)
        counter = 1
        while (cap.isOpened()):
            ret, frame = cap.read()
            # frame = padding.add_padding(frame, MAX_HEIGHT, MAX_WIDTH)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = padding.pad_and_resize(frame, YSIZE, XSIZE, gray=True)
            if frame is not None:
                if counter == middle_frame:
                    # cv.imshow('Testing', frame)						
                    image_list.append(frame)
                    cap.release()
                    # cv.waitKey(0)
            counter = counter + 1

    return np.asarray(image_list)