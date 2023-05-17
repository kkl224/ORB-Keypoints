
import matplotlib.pyplot as plt





def path_intervals(paths):
    intervals = [(path[0][0], path[-1][0]) for path in paths]
    return intervals
def kp_path(frame_kps, path, time_stop=-1):
    x, y = [], []
    for vertex in path:
        t, kp_id = vertex
        kp = frame_kps[t][kp_id]
        # print(kp.pt)
        x.append(kp.pt[0])
        y.append(kp.pt[1])

        if t >= time_stop:
            break
    return x, y


def plot_path(frame_kps, path):
    x, y = kp_path(frame_kps,path)
    plt.plot(x, y, '.-')


def plot_kp_timeline(paths):
    intervals = path_intervals(paths)
    for i, (start, end) in enumerate(intervals):
        plt.plot([start, end], [i, i])

    plt.title("Keypoint timeline")
    plt.xlabel("Frame number")
    plt.ylabel("Keypoint id")
    plt.grid()
    plt.show()