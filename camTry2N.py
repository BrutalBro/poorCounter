import cv2
import torch
import pafy
from numpy import random
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import copy
from PIL import ImageGrab
# from tracker import *

# url = 'https://youtu.be/5QEQzojkXKM'
url = 'https://youtu.be/xn7t5kQ21Es'
# url = 'https://youtu.be/wCcMcaiRbhM'
# url = 'https://youtu.be/7jW8r_Vkf5c'
video = pafy.new(url)
best = video.getbest(preftype="mp4")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
capture = cv2.VideoCapture(best.url)

lossCntFrame = 150
confTreshold = 0.35

peopleCounter = 0
trackers = dict()
trackersColors = dict()
trackersToDelete = dict()
prevDets = np.array([])
activeBoxes = {}

TrDict = {
	'csrt' : cv2.TrackerCSRT_create,
	'kcf' : cv2.TrackerKCF_create,
	'boosting' : cv2.TrackerBoosting_create,
	'mil' : cv2.TrackerMIL_create,
	'tld' : cv2.TrackerTLD_create,
	'medianflow' : cv2.TrackerMedianFlow_create,
	'mosse' : cv2.TrackerMOSSE_create,
}

def updateAllTrackers(frame):
	res = []
	for k in trackers.keys():
		t = trackers[k]
		res.append((k, t.update(frame)))

	return res

def addNewTrackers(boxes, frame):
	global peopleCounter
	for b in boxes:
		trackers[peopleCounter] = TrDict['csrt']()
		trackers[peopleCounter].init(frame, tuple(b))
		trackersColors[peopleCounter] = [random.randint(0, 255) for _ in range(3)]
		peopleCounter += 1

def tryAddTrackers(dets, trackBoxesRes, frame):
	newDets = []
	for d in dets:
		minD = 1e9
		maxIntersect = 0
		dc = (d[0] + d[2]/2, d[1] + d[3]/2)

		for t in trackBoxesRes:
			key, (_, b) = t
			tc = (b[0] + b[2]/2, b[1] + b[3]/2)

			if minD > (tc[0]-dc[0])**2 + (tc[0]-dc[0])**2:
				minD = (tc[0]-dc[0])**2 + (tc[0]-dc[0])**2

			xmin = max(d[0], b[0])
			ymin = max(d[1], b[1])

			xmax = min(d[0] + d[2], b[0] + b[2])
			ymax = min(d[1] + d[3], b[1] + b[3])
			if xmin < xmax and ymin < ymax and maxIntersect < (xmax-xmin)*(ymax-ymin):
				maxIntersect = (xmax-xmin)*(ymax-ymin)

		intersectRatio = maxIntersect/(b[2]*b[3])

		if intersectRatio == 0 and d[4] > 0.5: # and minD > 10:
			newDets.append(d)

	if len(newDets) > 0:
		addNewTrackers(np.array(newDets)[:,:-1], frame)
			

def tryRemoveTrackers(dets, trackBoxesRes, frame):
	global trackersToDelete
	for t in trackBoxesRes:
		key, (_, b) = t
		minD = 1e9
		maxIntersect = 0
		tc = (b[0] + b[2]/2, b[1] + b[3]/2)

		for d in dets:
			dc = (d[0] + d[2]/2, d[1] + d[3]/2)

			if minD > (tc[0]-dc[0])**2 + (tc[0]-dc[0])**2:
				minD = (tc[0]-dc[0])**2 + (tc[0]-dc[0])**2

			xmin = max(d[0], b[0])
			ymin = max(d[1], b[1])

			xmax = min(d[0] + d[2], b[0] + b[2])
			ymax = min(d[1] + d[3], b[1] + b[3])
			if xmin < xmax and ymin < ymax and maxIntersect < (xmax-xmin)*(ymax-ymin):
				maxIntersect = (xmax-xmin)*(ymax-ymin)

		intersectRatio = maxIntersect/(b[2]*b[3])
		print(f'{intersectRatio:.4}', t)

		if intersectRatio < 0.5:
			if key in trackersToDelete:
				trackersToDelete[key] += 1
			else:
				trackersToDelete[key] = 1
		else:
			trackers[key].init(frame, tuple(b))
			if key in trackersToDelete:
				trackersToDelete.pop(key)

	print('\n\n\n')

def removeTrackers():
	global trackersToDelete
	dk = []
	for k in trackersToDelete.keys():
		if trackersToDelete[k] == lossCntFrame:
			trackers.pop(k)
			dk.append(k)

	for k in dk:
		trackersToDelete.pop(k)
	

def plot_one_box(x, im, color=None, label=None, line_thickness=2):
	color = color or [0, 255, 0]
	c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	cv2.rectangle(im, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
	if label:
		tf = max(line_thickness - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(im, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# FRAMES = 4
cntF = 0
# while cntF != FRAMES:
# 	cntF += 1
while True:
	cntF += 1
	grabbed, frame = capture.read()
	# frame = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
	# frame = np.array(frame)
	# frame = cv2.resize(frame, (858, 480)) 

	t1 = time.time()
	res = model(frame)

	dets = np.array([])
	if len(res.pandas().xyxy) > 0:
		for vls in res.pandas().xyxy[0].values:
			xmin, ymin, xmax, ymax, conf, clss, name = vls

			if name == 'person' and conf > confTreshold:
				el = [xmin, ymin, xmax-xmin, ymax-ymin, conf]
				plot_one_box((xmin, ymin, xmax, ymax), frame, color=[0,0,0], label='det', line_thickness=1)
				dets = np.array([el]) if len(dets) == 0 else np.append(dets, [el], axis=0)

	print('dets:')
	for dd in dets:
		print(dd)
	print('end dets')
	if len(trackers) == 0:
		addNewTrackers(dets[:,:-1], frame)

	res = updateAllTrackers(frame)

	for colorKey, (_, box) in res:
		color = trackersColors[colorKey]
		xmin, ymin, w, h = box
		plot_one_box((xmin, ymin, xmin+w, ymin+h), frame, color=color, label=str(colorKey), line_thickness=1)

	tryRemoveTrackers(dets, res, frame)
	removeTrackers()
	tryAddTrackers(dets, res, frame)

	print(trackersToDelete)


	t2 = time.time()
	print(f'Done. ({t2 - t1:.3f}s)', peopleCounter)

	cv2.imshow('str(p)', frame)
	cv2.imwrite('RESULT/' + str(cntF) + '.jpg', frame)
	cv2.waitKey(1)  # 1 millisecond

	# if cntF == 4:
	# 	break


capture.release()
cv2.DestryAllWindows()


# Used links:

# https://github.com/cfotache/pytorch_objectdetecttrack/blob/master/PyTorch_Object_Tracking.ipynb (just was looked)
# https://jbencook.com/simple-pytorch-object-tracking/
# https://arxiv.org/ftp/arxiv/papers/1709/1709.08761.pdf


# sources:
# https://www.youtube.com/channel/UCpk2ftN35L3xfoV2S5xLN2A/videos
# https://www.youtube.com/watch?v=7jW8r_Vkf5c