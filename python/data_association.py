import cv2
import numpy as np
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment


@jit
def iou(boxA, boxB):
    xx1 = np.maximum(boxA[0], boxB[0])
    yy1 = np.maximum(boxA[1], boxB[1])
    xx2 = np.minimum(boxA[2], boxB[2])
    yy2 = np.minimum(boxA[3], boxB[3])

    interArea = max(0, xx2 - xx1 + 1) * max(0, yy2 - yy1 + 1)

    boxA_Area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_Area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou_value = interArea / float(boxA_Area + boxB_Area - interArea)

    return iou_value


def associate_detection_to_tracker(detections, trackers, iou_threshold=0.3):
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    print("iou_matrix: ",iou_matrix)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detection = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detection.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detection.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detection), np.array(unmatched_trackers)
