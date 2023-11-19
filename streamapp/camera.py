import torch
from pathlib import Path
import cv2
import math
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request,math
import numpy as np

from django.conf import settings


model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', os.path.join(settings.BASE_DIR,'weights/model.pt'))



classNames = ["with_mask", "without_mask","properly_not_weared"]


class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(1)

	def __del__(self):
		self.video .release()

	def get_frame(self):
		success, image = self.video.read()
	
		frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


		results=model(frame)
		if results.xyxy[0].shape[0] > 0:
		
			boxes = results.xyxy[0][:, :4].cpu().numpy()
			scores = results.xyxy[0][:, 4].cpu().numpy()
			labels = results.xyxy[0][:, 5].cpu().numpy()

			(x1, y1, x2, y2) = map(int, boxes[0])

			conf = math.ceil((scores[0] * 100)) / 100
			cls = int(labels[0])  # cls means the class_label either 0 or 1.
			print(".....cls",cls)

			class_name = classNames[cls]

			print("clsss",class_name)
			label = f'{class_name}:{conf}'

			t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
			c2 = x1 + t_size[0], y1 - t_size[1] - 3

			cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
			cv2.rectangle(image, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
			cv2.putText(image, label, (x1, y1 - 2), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)

			
		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()

