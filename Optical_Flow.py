import cv2 as cv
import numpy as np

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
        counter = 1
        stacked_frames = []

        while (cap.isOpened()):
            ret, frame = cap.read()
            if frame is not None:
                # cv.imshow("input", frame)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #gray scale of the next frame

                if counter >= middle_frame - 10 and counter <= middle_frame + 9:
                    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    horz = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
                    vert = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)
                    horz = horz.astype('uint8')
                    vert = vert.astype('uint8')
                    # cv.imshow('Horizontal Component', horz)
                    # cv.imshow('Vertical Component', vert)
                    stacked_frames.append(horz)
                    stacked_frames.append(vert)

                # Updates previous frame
                prev_gray = gray
                counter = counter + 1
            else:
                cap.release()

        frame_stack = np.asarray(stacked_frames)
        frame_stack = np.transpose(frame_stack, (1,2,0))
        stacked_videos.append(frame_stack)

    return stacked_videos