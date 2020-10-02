# python mark_preparingInputs_det2.py

import os
import numpy as np
import cv2
import random
import itertools
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#########################################    Start Codes    #########################################

# --- Directories
data_type = "train" #val #test
COCO_DIR = "/media/farhat/Farhat_SSD/MarkNET" + "/data/coco/" 

# --- Load Bounding Boxes
prepapred_data = np.load(COCO_DIR + data_type + '/marks_point_' + data_type + '.npz', allow_pickle=True)
bbox = list(prepapred_data['bbox'])
paths = list(prepapred_data['paths'])
paths = [path.replace("/train_person_area_face_masks/", '/train_person_area_face_masks_marks/') for path in paths]
    

# --- Test Annotations
def annotate_image(pth, marks):
    img = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)
    for mark in marks:
        x, y, h, w = mark
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

aList = [i for i in range(len(paths))]
for i in random.sample(aList, 10):
    img = annotate_image(paths[i], bbox[i])
    cv2.imshow('ImageWindow', img)
    cv2.waitKey()


# --- Extract Annotations
dataset = []
for index in tqdm(range(0, len(paths))):
    for mark in bbox[index]:
        data = {}
        img = cv2.cvtColor(cv2.imread(paths[index]), cv2.COLOR_BGR2RGB)
        width = img.shape[1]
        height =img.shape[0]
        x, y, h, w = mark
        data['file_name'] = paths[index]
        data['width'] = width
        data['height'] = height
        data["x_min"] = int(round(x))
        data["y_min"] = int(round(y))
        if(round(x+w) > width): data["x_max"] = int(round(width))
        else: data["x_max"] = int(round(x+w))
        if(round(y+h) > height): data["y_max"] = int(round(height))
        else: data["y_max"] = int(round(y+h))
        data['class_name'] = 'mark'
        dataset.append(data)

df = pd.DataFrame(dataset)
df.to_csv(COCO_DIR + "/marks_annotations.csv", header=True, index=None)

# --- Train-Test Splitting
unique_files = df.file_name.unique()
train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.95), replace=False))
train_df = df[df.file_name.isin(train_files)]
test_df = df[~df.file_name.isin(train_files)]

train_df.to_csv(COCO_DIR + "/marks_annotations_train.csv", header=True, index=None)
test_df.to_csv(COCO_DIR + "/marks_annotations_test.csv", header=True, index=None)

print("Finished Prepartion...")
