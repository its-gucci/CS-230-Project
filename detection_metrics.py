from darkflow.net.build import TFNet
import cv2
import os 
import json

def detrac_names_loader(path):
	image_names = []
	for file in os.listdir(path):
		if file.endswith(".jpg"):
			image_names.append(path + file)
	return image_names

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((yi2 - yi1), 0) * max((xi2 - xi1), 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area/union_area
    
    return iou

def xml_parser():
	"""
	Given darkflow xml annotations, return list of objects and their bounding boxes
	"""
	return

def json_parser(json_object):
	"""
	Given darkflow json output, returns list of tuples (predicted_label, bounding_box)
	"""
	loaded = json.loads(json_object)
	predictions = []
	for i in range(len(loaded)):
		prediction = loaded[i]
		predict_label = prediction["label"]
		predict_box = (prediction["topleft"]["x"], prediction["topleft"]["y"], prediction["bottomright"]["x"], prediction["bottomright"]["y"])
		predictions.append(predicted_label, predict_box)
	return predictions

def main():
	options = {"model": "cfg/yolo.cfg", "load": "-1", "threshold": 0.5}

	tfnet = TFNet(options)

	images = detrac_loader(file_dir)

	for image_name in images:
		imgcv = cv2.imread(image_name)
		result = tfnet.return_predict(imgcv)
		

if __name__ == "__main__": 
	main()