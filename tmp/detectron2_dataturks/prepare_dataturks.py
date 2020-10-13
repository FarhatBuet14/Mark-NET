# python prepare_data.py --dwn True
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import urllib.request as ul
import json
import PIL.Image as Image
import matplotlib.pyplot as plt
import argparse

from utills import annotate_image

# --- Parse the arguements
parser = argparse.ArgumentParser(description='Necessary variables')
parser.add_argument("--dwn",type=bool, default=False, help = "Download the dataset from online")
arguments = parser.parse_args()

data_dir = "./data/Dataturks/"
IMAGES_PATH = data_dir + 'faces'
faces_df = pd.read_json(data_dir + 'face_detection.json', lines=True)
if(arguments.dwn): os.makedirs(data_dir + "faces", exist_ok=True)

dataset = []
if(arguments.dwn): print("Downloading...")
else: print("Reading Annotations...")
for index, row in tqdm(faces_df.iterrows(), total=faces_df.shape[0]):
	image_name = f'face_{index}.jpeg'
	if(arguments.dwn): img = ul.urlopen(row["content"])
	else: img = f'{IMAGES_PATH}/{image_name}'

	img = Image.open(img)
	img = img.convert('RGB')

	if(arguments.dwn): img.save(data_dir + f'faces/{image_name}', "JPEG")

	annotations = row['annotation']

	for an in annotations:
		data = {}

		width = an['imageWidth']
		height = an['imageHeight']
		points = an['points']

		data['file_name'] = image_name
		data['width'] = width
		data['height'] = height

		data["x_min"] = int(round(points[0]["x"] * width))
		data["y_min"] = int(round(points[0]["y"] * height))
		data["x_max"] = int(round(points[1]["x"] * width))
		data["y_max"] = int(round(points[1]["y"] * height))

		data['class_name'] = 'face'
		dataset.append(data)

df = pd.DataFrame(dataset)
print("Data file, Annotation")
print(df.file_name.unique().shape[0], df.shape[0])

df.to_csv(data_dir + 'annotations.csv', header=True, index=None)

# Check Annotation
img_df = df[df.file_name == df.file_name.unique()[0]]
img = annotate_image(IMAGES_PATH, img_df, resize=False)

plt.imshow(img)
plt.axis('off')
