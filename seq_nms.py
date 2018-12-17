import numpy as np
#cimport numpy as np
#cimport cython
#from libc.math cimport exp
from ..utils.box import BoundBox

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((yi2 - yi1), 0) * max((xi2 - xi1), 0)
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###
    
    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area/union_area
    ### END CODE HERE ###
    
    return iou

def convert_box(x, y, w, h):
    return (x - w/2, y - h/2, x + w/2, y + h/2)

#Seq-NMS
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
#@cython.cdivision(True)
def Seq_NMS(final_probs, final_bbox, rescoring="avg"):
    """
    Inputs:
    final_probs: final_probs[t] are the probabilities for the bounding boxes of the t-th frame of the video
    final_bbox: final_bbox[t] are the raw bounding boxes for the t-th frame of the video

    Returns: list of bounding box predictions for the whole video, each element is the list of BoundBox objects for that frame
    """
    # number of frames in the video
    T = len(final_probs)
    # number of bounding boxes per frame
    num_boxes = final_bbox[0].shape[0]
    num_classes = final_probs[0].shape[1]
    print("Boxes per frame: {}".format(num_boxes))
    print("Number of classes: {}".format(num_classes))

    # keeps track of the suppressed indices
    suppressed_indices = set()

    # keeps track of the predicted bounding boxes for each frame in the video
    predicted_boxes = [list([]) for _ in range(T)]

    #Seq-NMS algorithm
    while T * num_boxes - len(suppressed_indices) > 0:
        # Initialize score array and previous for dynamic programming
        best_scores = np.zeros((T, num_boxes))
        previous_index = np.full((T, num_boxes), -1)

        # update scores using DP
        overall_max = 0
        overall_indices = (-1, -1)
        for t in range(T):
            if t == 0:
                for i in range(num_boxes):
                    if (t, i) in suppressed_indices:
                        continue
                    best_scores[t][i] = final_bbox[t][i, 4]
                    if final_bbox[t][i, 4] > overall_max:
                        overall_max = final_bbox[t][i, 4]
                        overall_indices = (t, i)
            else:
                for i in range(num_boxes):
                    # if this cell has been suppressed, continue
                    if (t, i) in suppressed_indices: 
                        continue
                    box_i = convert_box(final_bbox[t][i, 0], final_bbox[t][i, 1], final_bbox[t][i, 2], final_bbox[t][i, 3])
                    max_score = 0
                    max_scoring_box_idx = -1
                    # Find maximum box_j with IOU(box_i, box_j) > 0.5
                    for j in range(num_boxes):
                        # if this cell has been suppressed, continue
                        if (t - 1, j) in suppressed_indices:
                            continue
                        box_j = convert_box(final_bbox[t - 1][j, 0], final_bbox[t - 1][j, 1], final_bbox[t - 1][j, 2], final_bbox[t - 1][j, 3])
                        # Adjacent frames must have IoU > 0.5
                        if iou(box_i, box_j) > 0.5:
                            if best_scores[t - 1][j] > max_score:
                                max_score = best_scores[t - 1][j]
                                max_scoring_box_idx = j
                    if max_scoring_box_idx == -1:
                        best_scores[t][i] = final_bbox[t][i, 4]
                    else:
                        best_scores[t][i] = max_score + final_bbox[t][i, 4]
                        previous_index[t][i] = max_scoring_box_idx
                    if best_scores[t][i] > overall_max:
                        overall_max = best_scores[t][i]
                        overall_indices = (t, i)

        # find maximal sequence 
        sequence = [overall_indices]
        curr_idx = overall_indices
        while curr_idx[0] > 0 and previous_index[curr_idx[0]][curr_idx[1]] != -1:
            curr_idx = (curr_idx[0] - 1, previous_index[curr_idx[0]][curr_idx[1]])
            sequence = [curr_idx] + sequence

        # in a video, the object should be present for more than 1 frame
        if len(sequence) == 1:
            break

        print("Current sequence: {}".format(sequence))

        # Rescoring
        if rescoring == "avg":
            # replace the score of each box with the avg score of the sequence
            total = 0            
            for (t, i) in sequence:
                total += final_bbox[t][i, 4]
            avg = float(total)/len(sequence)
            for (t, i) in sequence:
                final_bbox[t][i, 4] = avg
        elif rescoring == "max":
            # replace the score of each box with the max score in the sequence
            max_score = 0
            for (t, i) in sequence:
                if final_bbox[t][i, 4] > max_score:
                    max_score = final_bbox[t][i, 4]
            for (t, i) in sequence:
                final_bbox[t][i, 4] = max_score

        # Suppression
        for (t, i) in sequence:
            box_i = convert_box(final_bbox[t][i, 0], final_bbox[t][i, 1], final_bbox[t][i, 2], final_bbox[t][i, 3])
            for j in range(num_boxes):
                box_j = convert_box(final_bbox[t][j, 0], final_bbox[t][j, 1], final_bbox[t][j, 2], final_bbox[t][j, 3])
                # suppress boxes in the same frame if the IoU > 0.3
                if iou(box_i, box_j) > 0.3:
                    suppressed_indices.add((t, j))

        # add sequence to predicted sequences
        for (t, i) in sequence:
            bb = BoundBox(num_classes)
            bb.x = final_bbox[t][i, 0]
            bb.y = final_bbox[t][i, 1]
            bb.w = final_bbox[t][i, 2]
            bb.h = final_bbox[t][i, 3]
            bb.c = final_bbox[t][i, 4]
            bb.probs = final_probs[t][i, :]
            #print("YOLO class probabilities: {}".format(bb.probs))
            predicted_boxes[t].append(bb)

    return predicted_boxes