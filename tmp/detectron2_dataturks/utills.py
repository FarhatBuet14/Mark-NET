import numpy as np 
import cv2
import itertools

def annotate_image(data_dir, annotations, resize=True):
	
	file_name = annotations.file_name.to_numpy()[0]
	img = cv2.cvtColor(cv2.imread(f'{data_dir}/{file_name}'), cv2.COLOR_BGR2RGB)
	for i, a in annotations.iterrows():
		cv2.rectangle(img, (a.x_min, a.y_min), (a.x_max, a.y_max), (0, 255, 0), 2)
	
	if not resize: return img

	return cv2.resize(img, (384, 384), interpolation = cv2.INTER_AREA)
