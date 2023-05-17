import cv2
import matplotlib.pyplot as plt

from timeline_generator import find_timelines_from_video, bf, orb, time_descriptor
import numpy as np

def keypoints_from_image_file(image_file):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors in the frame
    keypoints, des = orb.detectAndCompute(img, None)
    print(keypoints)

    # img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display the image with keypoints
    # plt.imshow(img_with_keypoints)
    # plt.show()

    return keypoints, des, img






def find_times(time_kps):
    times = []
    for s, e, kpts, dess in time_kps:
        if not s in times:
            times.append(s)
        if not e in times:
            times.append(e)

    return sorted(times)

def descriptors_per_interval(times, time_kps, frame_kps, frame_des, plot=False):
    # For each interval in a sequence, find the keypoints
    intervals_in_sequence = []
    for s1, e1 in zip(times[:-1], times[1:]):

        interval_descriptors = []
        interval_kpts = []
        # check if the keypoints are in the interval
        for s2, e2, kpts, dess in time_kps:
            # check if the intervals overlap
            if e1 >= s2 and e2 >= s1:
                interval_descriptors.append(dess)
                interval_kpts.append(kpts)
        # intervals_in_sequence.append((s1, e1, interval_kpts, np.array(interval_descriptors))) ##TODO
        intervals_in_sequence.append((s1, e1, frame_kps[s1], frame_des[s1]))
        # print(s1, e1, len(interval_descriptors))

    if plot:
        plt.plot([len(dess) for s, e, kpts, dess in intervals_in_sequence])
        plt.title("Number of Keypoints per interval")
        plt.show()

    return intervals_in_sequence


if __name__ == "__main__3":
    long_paths, frame_kps, frame_des, frames = find_timelines_from_video()

    f1, f2 = 20, 0
    img1, img2 = frames[f1], frames[f2]
    des1, des2 = frame_des[f1], frame_des[f2]
    kps1, kps2 = frame_kps[f1], frame_kps[f2]

    best_match = bf.match(des1, des2)
    frame_matches = cv2.drawMatches(img1, kps1, img2, kps2, best_match, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Plot best
    plt.imshow(frame_matches)
    for match in best_match:
        # Get the keypoints from the matches
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kps1[img1_idx].pt
        (x2, y2) = kps2[img2_idx].pt

        # Draw a line between the keypoints with thicker line width
        plt.plot([x1, x2 + img1.shape[1]], [y1, y2], '.-',linewidth=1, alpha=0.8)

    print("Number of matches ", len(best_match), " Kps1=",len(kps1), "Kps2=", len(kps2) )
    plt.show()


if __name__ == "__main__":
    # obtain keypoint timeline
    long_paths, frame_kps, frame_des, frames = find_timelines_from_video()
    time_kps = [time_descriptor(path, frame_kps, frame_des) for path in long_paths]

    # print(times)
    times = find_times(time_kps)
    intervals_in_sequence = descriptors_per_interval(times, time_kps, frame_kps, frame_des)

    # Read image
    img_keypoints, img_descriptors, img = keypoints_from_image_file('media/3.jpg')
    # img_keypoints, img_descriptors, img = frame_kps[20], frame_des[20], frames[20]

    # find the interval that matches the most with the image descriptors
    sum_distances = []
    best_match = None
    best_interval_id = -1
    best_sq_dist = 100000000
    best_kpts = None
    best_des = None
    for i, (s, e, kpts, descriptors) in enumerate(intervals_in_sequence):
        matches = bf.match(img_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # print(s, e, len(matches))
        sum_distance = sum([m.distance for m in matches]) / len(matches)
        sum_distances.append(sum_distance)

        if sum_distance <= best_sq_dist:
            best_match = matches
            best_sq_dist = sum_distance
            best_interval_id = i
            best_kpts = kpts
            best_des = descriptors
            print("len best match", len(best_match))


    # print(sum_distances)


    x = np.hstack([(s,e) for (s, e, kpts, img_descriptors) in intervals_in_sequence])
    y = np.hstack([(dist,dist) for dist in sum_distances])
    plt.plot(x, y, '-')
    for (s, e, kpts, img_descriptors), dist in zip(intervals_in_sequence, sum_distances):
        x = [s,e]
        # print(x)
        y = [dist, dist]
        plt.plot(x,y)

    # plt.plot(sum_distances)
    plt.title("Average distance of descriptors in interval")
    plt.show()

    # best interval
    id_max_interval = np.argmin(sum_distances)
    bs, be, bkpts, bdess = intervals_in_sequence[id_max_interval]


    best_frame = frames[bs]
    # best_kpts = frame_kps[bs]
    # bdess = frame_des[bs]
    # best_kpts = frame_kps[bs]
    # best_match = bf.match(img_descriptors, bdess)
    print('Number of best matches', len(best_match), " time=", bs)

    best_match = sorted(best_match, key=lambda x: x.distance)[:30]

    frame_matches = cv2.drawMatches(img, img_keypoints, best_frame, best_kpts, best_match, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



    # Plot best
    plt.imshow(frame_matches)
    for match in best_match:
        # Get the keypoints from the matches
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = img_keypoints[img1_idx].pt
        (x2, y2) = best_kpts[img2_idx].pt

        # Draw a line between the keypoints with thicker line width
        plt.plot([x1, x2 + img.shape[1]], [y1, y2], linewidth=2, alpha=0.8)

    # Display the frame with matches
    # cv2.imshow('Frame with matches', frame_matches)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # break
    plt.show()
    # print(len(time_kps))

