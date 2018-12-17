import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from ...cython_utils.box_constructor_snms import box_constructor_snms

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out, video=False):
	# meta
	meta = self.meta
	boxes = list()
	if not video:
		boxes=box_constructor(meta, net_out)
	else:
		boxes=box_constructor_snms(meta, net_out)
	return boxes

#def findboxes_snms(self, net_out):
#	# meta
#	meta = self.meta
#	boxes = list()
#	boxes=box_constructor_snms(meta, net_out)
#	return boxes

def postprocess(self, net_out, im, save = True, video=False):
	"""
	Takes net output, draw net_out, save to disk
	"""
	if not video:
		boxes = self.findboxes(net_out)

		# meta
		meta = self.meta
		threshold = meta['thresh']
		colors = meta['colors']
		labels = meta['labels']
		if type(im) is not np.ndarray:
			imgcv = cv2.imread(im)
		else: imgcv = im
		h, w, _ = imgcv.shape
	
		resultsForJSON = []
		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			thick = int((h + w) // 300)
			if self.FLAGS.json:
				resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
				continue

			cv2.rectangle(imgcv,
				(left, top), (right, bot),
				colors[max_indx], thick)
			cv2.putText(imgcv, mess, (left, top - 12),
				0, 1e-3 * h, colors[max_indx],thick//3)

		if not save: return imgcv

		outfolder = os.path.join(self.FLAGS.imgdir, 'out')
		img_name = os.path.join(outfolder, os.path.basename(im))
		if self.FLAGS.json:
			textJSON = json.dumps(resultsForJSON)
			textFile = os.path.splitext(img_name)[0] + ".json"
			with open(textFile, 'w') as f:
				f.write(textJSON)
			return

		cv2.imwrite(img_name, imgcv)

	if video:
		box_seq = self.findboxes(net_out, video=True)
		#print("Checking length of our bounding box predictions: {}".format(len(box_seq)))
		for i in range(len(box_seq)):
			#print("Current image number: {}".format(i))
			boxes = box_seq[i]
			image = im[i]

			# meta
			meta = self.meta
			threshold = meta['thresh']
			colors = meta['colors']
			labels = meta['labels']
			if type(image) is not np.ndarray:
				imgcv = cv2.imread(image)
			else: imgcv = image
			h, w, _ = imgcv.shape

			#print("predict.py current image: {}".format(image))
			#print("Threshold: {}".format(threshold))
			#print(boxes)
			resultsForJSON = []
			for b in boxes:
				max_indx = np.argmax(b.probs)
				#print("Best class: {}".format(max_indx))
				max_prob = b.probs[max_indx]
				#print("Highest probability: {}".format(max_prob))
				boxResults = self.process_box(b, h, w, threshold)
				if boxResults is None:
					continue
				left, right, top, bot, mess, max_indx, confidence = boxResults
				thick = int((h + w) // 300)
				if self.FLAGS.json:
					resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
					continue

				cv2.rectangle(imgcv,
					(left, top), (right, bot),
					colors[max_indx], thick)
				cv2.putText(imgcv, mess, (left, top - 12),
					0, 1e-3 * h, colors[max_indx],thick//3)

			if not save: return imgcv

			outfolder = os.path.join(self.FLAGS.imgdir, 'out')
			img_name = os.path.join(outfolder, os.path.basename(image))
			if self.FLAGS.json:
				textJSON = json.dumps(resultsForJSON)
				textFile = os.path.splitext(img_name)[0] + ".json"
				with open(textFile, 'w') as f:
					f.write(textJSON)
				continue

			cv2.imwrite(img_name, imgcv)