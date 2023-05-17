import math
from time import sleep
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cv2 import norm

from util import path_intervals, plot_kp_timeline
from video_generator import save_kp_video, extract_keypoints

import networkx as nx

WINDOW_T = 15

VIDEO_FILE = 'media/video0.mp4'

MAX_FRAMES = 1000  # large numbers will cover the whole video
SHORTEST_LENGTH = 5  # min 5
MAX_MATCH_DISTANCE = 15  # less than 50 is ok

# Create an ORB object and detect keypoints and descriptors in the template
orb = cv2.ORB_create()
# Create a brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def find_best_matches(frame_des1, frame_des2):
    # matches(query_this_image, train)
    matches = bf.match(frame_des1, frame_des2)
    # Sort the matches by distance
    # matches = sorted(matches, key=lambda x: x.distance)

    # Best matches: Filter by max distance
    matches = [m for m in matches if m.distance < MAX_MATCH_DISTANCE]
    best_edges = [(m.queryIdx, m.trainIdx) for m in matches]

    return best_edges


def sequential_matches_graph(frame_des, T=1):
    edges = []  # ((time, kp_id), time, kp_id)
    for i in range(len(frame_des) - T):
        for t in range(1, T + 1):
            best_edges = find_best_matches(frame_des[i], frame_des[i + t])
            for e in best_edges:
                edges.append(((i, e[0]), (i + t, e[1])))

    # Create an empty graph
    G = nx.DiGraph()
    # Add the edges to the graph
    G.add_edges_from(edges)

    # TODO: Remove edges if the keypoint jumps
    return G


def find_long_paths(G):
    # Add new nodes X and Z
    x = 0
    z = -1
    G.add_nodes_from([x, z])

    # Connect X to all nodes without a predecessor
    no_pred = [n for n in G.nodes() if G.in_degree(n) == 0]
    for n in no_pred:
        G.add_edge(x, n)

    # Connect Z to all nodes without a successor
    no_succ = [n for n in G.nodes() if G.out_degree(n) == 0]
    for n in no_succ:
        G.add_edge(n, z)

    # Compute all simple paths in the graph
    all_paths = list(nx.all_simple_paths(G, x, z))
    all_paths = sorted(all_paths, key=lambda x: -len(x))

    # for p in all_paths:
    #     print("path", len(p))
    # # Print the paths
    print("All paths:", len(all_paths))

    # Find the paths with length greater than a threshhold. remove the source and sink nodes.
    long_paths = [p[1:-1] for p in all_paths if len(p) >= SHORTEST_LENGTH]
    print("Long paths:", len(long_paths))
    return long_paths


def find_long_paths_T(G):
    # Invert the graph
    G_inv = nx.DiGraph()
    G_inv.add_edges_from([(v, u) for u, v in G.edges])

    visited_flag = {v: 0 for v in G.nodes}
    paths = []
    # Run topological sort
    sorted_vertices = list(nx.topological_sort(G_inv))
    # for each subtree
    for v in sorted_vertices:
        # v was already visited?
        if visited_flag[v]:
            continue

        v_decendents = nx.descendants(G_inv, v)

        oldest_t = math.inf
        oldest_v = None
        for u in v_decendents:
            # was a decendent visited?
            # if visited_flag[u]:
            #     break

            # mark as visited
            visited_flag[u] = 1

            # find the oldest
            t, _ = u
            if t < oldest_t:
                oldest_t = t
                oldest_v = u

        if oldest_v is None:
            continue

        path = nx.shortest_path(G, source=oldest_v, target=v)  # FIXME Switch to longest path
        paths.append(path)

    # Find the paths with length greater than a threshhold. remove the source and sink nodes.
    long_paths = [p for p in paths if len(p) >= SHORTEST_LENGTH]
    print("Long paths:", len(long_paths))

    # sort by frame time
    long_paths = sorted(long_paths, key=lambda x: x[0])
    return long_paths


def time_descriptor(path, frame_kpts, frame_des, best_descriptor=False):
    start = path[0][0]
    end = path[-1][0]
    #TODO include x and y
    descriptor = None

    min_distance = math.inf
    min_descriptor = None

    if best_descriptor:
        # Compute the descriptor that can represent the whole path
        for t, kp_id in path:
            des1 = frame_des[t][kp_id]
            sqr_dis = 0  # sum of square distances
            for t2, kp_id2 in longest_path:
                des2 = frame_des[t2][kp_id2]
                sqr_dis += cv2.norm(des1, des2, cv2.NORM_HAMMING) ** 2
            if sqr_dis < min_distance:
                min_descriptor = des1
                min_distance = sqr_dis

        descriptor = min_descriptor
    else:
        # first element in path
        t, kp_id = path[0]
        descriptor = frame_des[t][kp_id]
        kpts = frame_kpts[t][kp_id]

    return start, end, kpts, descriptor


def find_timelines_from_video(save_video=False):
    frame_kps, frame_des, frames = extract_keypoints(VIDEO_FILE, max_frames=MAX_FRAMES)
    G = sequential_matches_graph(frame_des, T=WINDOW_T)
    # Print some information about the graph
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # Find the long paths that pass a threshold
    long_paths = find_long_paths_T(G)

    # Save video as a mp4 file
    if save_video:
        save_kp_video(frames, frame_kps, long_paths)

    # triplets (t_start, t_end, descriptor), per path
    return long_paths, frame_kps, frame_des, frames


if __name__ == "__main__":
    long_paths, frame_kps, frame_des, frames = find_timelines_from_video(save_video=True)
    time_kps = [time_descriptor(path, frame_kps, frame_des) for path in long_paths]
    # ## Show timeline
    plot_kp_timeline(long_paths)

    # Longest path
    id_max = np.argmax([path[-1][0] - path[0][0] for path in long_paths])
    longest_path = long_paths[id_max]
    print("Longest path:", longest_path[-1][0] - longest_path[0][0], " Nodes ", len(longest_path))
    print(longest_path)
    # long_paths=[longest_path]

# for tkps in time_kps:
#     print(tkps.start,tkps.end)

# # Longest path analysis
# d = []
# for t, kp_id in longest_path:
#     distance = 0
#     ld = []
#     for t2, kp_id2 in longest_path:
#         des1 = frame_des[t][kp_id]
#         des2 = frame_des[t2][kp_id2]
#         norm = cv2.norm(des1, des2, cv2.NORM_HAMMING)
#         distance+=norm**2
#         # print((t,kp_id), (t2, kp_id2), distance)
#         ld.append(norm)
#     d.append(distance)
#     plt.plot(ld)
# plt.show()
#
# plt.plot(d)
# plt.show()


# ## Show paths
# for path in long_paths:
#     plot_path(frame_kps, path)
# plt.show()
#
#

# print(edges)


# print("Nodes:", list(G.nodes()))
# print("Edges:", list(G.edges()))

#
# nx.draw(G, with_labels=True)
# plt.show()


# # # Find the longest path in the graph
# longest_path = nx.dag_longest_path(G)
# #
# # # Print the longest path
# print("Longest path:", len(longest_path), longest_path)
#
