import numpy as np
from .seq_nms import Seq_NMS

def expit(x):
	return 1. / (1. + np.exp(-x))

def box_constructor_snms(meta, net_out_in):

    threshold = meta['thresh']
    anchors = np.asarray(meta['anchors'])

    H, W, _ = meta['out_size']
    C = meta['classes']
    B = meta['num']

    final_probs = []
    final_bbox = []

    for t in range(len(net_out_in)):
        net_out = net_out_in[t].reshape([H, W, B, int(net_out_in[t].shape[2]/B)])
        Classes = net_out[:, :, :, 5:]
        Bbox_pred =  net_out[:, :, :, :5]
        probs = np.zeros((H, W, B, C))

        # For each frame, recover the bounding box info and probability info as usual
        for row in range(H):
            for col in range(W):
                for box_loop in range(B):
                    arr_max = 0
                    arr_sum = 0
                    Bbox_pred[row, col, box_loop, 4] = expit(Bbox_pred[row, col, box_loop, 4])
                    Bbox_pred[row, col, box_loop, 0] = (col + expit(Bbox_pred[row, col, box_loop, 0])) / W
                    Bbox_pred[row, col, box_loop, 1] = (row + expit(Bbox_pred[row, col, box_loop, 1])) / H
                    Bbox_pred[row, col, box_loop, 2] = np.exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2 * box_loop + 0] / W
                    Bbox_pred[row, col, box_loop, 3] = np.exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 1] / H
                    #SOFTMAX BLOCK, no more pointer juggling
                    for class_loop in range(C):
                        arr_max=max(arr_max,Classes[row,col,box_loop,class_loop])
                
                    for class_loop in range(C):
                        Classes[row,col,box_loop,class_loop]=np.exp(Classes[row,col,box_loop,class_loop]-arr_max)
                        arr_sum+=Classes[row,col,box_loop,class_loop]
                
                    for class_loop in range(C):
                        tempc = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]/arr_sum
                        #print("tempc: {}".format(tempc))
                        probs[row, col, box_loop, class_loop] = tempc
                        #print("Probs has changed: {}".format(probs[row, col, box_loop, :]))

        # Append each frame's bounding boxes and probabilities to the list for the whole video
        #print("Predicted class probabilities: {}".format(probs))
        final_probs.append(np.ascontiguousarray(probs).reshape(H*W*B,C))
        final_bbox.append(np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5))

    #print("Checking length of image probabilities: {}".format(len(final_probs)))
    #print("Checking length of image bounding boxes: {}".format(len(final_bbox)))
    
    #Seq-NMS                    
    return Seq_NMS(final_probs, final_bbox)
