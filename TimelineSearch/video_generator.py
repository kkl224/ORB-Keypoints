
import numpy as np
import cv2

from util import path_intervals, kp_path


def time_in_path(time, path, frame_kps):
    for vertex in path:
        t, kp_id = vertex

        if t >= time:
            kp = frame_kps[t][kp_id]
            # print(kp.pt)
            px = kp.pt[0]
            py = kp.pt[1]
            return px, py



def extract_keypoints(video, max_frames):
    # Create a VideoCapture object to read the video file
    cap = cv2.VideoCapture(video)
    # Extract all keypoints and descriptors by frame
    frame_kps, frame_des = [], []
    video_frames = []
    k = 0

    # Create an ORB object and detect keypoints and descriptors in the template
    orb = cv2.ORB_create()


    # Loop through the video frames
    while cap.isOpened() and k < max_frames:

        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if ret:
            # Display the frame
            # cv2.imshow('Video Frame', frame)
            # Wait for a key press and check if the 'q' key was pressed
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

            # Detect keypoints and compute descriptors in the frame
            kp2, des2 = orb.detectAndCompute(frame, None)

            if des2 is None:
                print("No keypoints/descriptors in frame ", k)
                continue


            frame_kps.append(kp2)
            frame_des.append(des2)
            video_frames.append(frame)
            # print(k, " keypoints: ", len(kp2))

            k += 1
        else:
            # Exit the loop if the end of the video is reached
            break
    # Release the VideoCapture object and close all windows
    cap.release()
    return frame_kps, frame_des, video_frames
    # cv2.destroyAllWindows()


def save_kp_video(frames, frame_kps, kp_paths):
    intervals = path_intervals(kp_paths)

    color_paths= [tuple(map(int, np.random.randint(0, 255, size=3))) for _ in kp_paths]

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width  = len(frames[0][0])
    height = len(frames[0])
    print(width,height)
    fps = 10
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))


    for t, frame in enumerate(frames):
        for i, (start, end) in enumerate(intervals):
            if not (start <= t <= end):
                continue

            path = kp_paths[i]
            x, y = kp_path(frame_kps, path, t)
            cv_path = np.array([point for point in zip(x,y)], dtype=np.int32)

            # Generate a random color for the polyline
            color = color_paths[i]
            # Draw
            cv2.polylines(frame, [cv_path], isClosed=False, color=color, thickness=2)

            point = np.array(time_in_path(t, path, frame_kps), dtype=np.int32)
            cv2.circle(frame, point, 5, color, -1)



        # cv2.imshow('Video Frame', frame)
        out.write(frame)

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        #
        # sleep(.1)

    out.release()