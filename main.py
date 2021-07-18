import requests
import cv2
import numpy as np
import cvlib as cv

from cvlib.object_detection import draw_bbox


url = "http://192.168.2.246:8080/shot.jpg"


print('AI CAMERA Starting....')

while True:
	img_resp = requests.get(url)
	img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
	img = cv2.imdecode(img_arr, -1)
	  


	bbox, label, conf = cv.detect_common_objects(img)

	output_image = draw_bbox(img,bbox,label,conf)

	

	
	
	cv2.imshow("AI CAMERA", img)

	if cv2.waitKey(1) == 27:
		break