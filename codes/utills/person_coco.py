# python person_coco.py
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import shutil
from tqdm import tqdm
import pandas as pd
from collections import Counter

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


# To find local version - https://github.com/matterport/Mask_RCNN
# sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))
# sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))
from coco import coco
config = coco.CocoConfig()
data_type = "val" #train #test
COCO_DIR = "/media/farhat/Farhat_SSD/MarkNET" + "/data/coco/" + data_type

data = pd.read_csv(COCO_DIR + "/annotations/1person_IDs.csv")
person_ids = list(data.iloc[:, 1].values)

# Load dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, subset= data_type, year="2017", class_ids = [1], image_ids=person_ids)

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))

#--- 1-Person Selection 
# person_count = 0
# person_ids = []
# img_ids = []
# for i in tqdm(range(len(dataset.image_ids))):
#     image_id = dataset.image_ids[i]
#     _, class_ids = dataset.load_mask(image_id)
#     if(Counter(class_ids)[1]==1): 
#         person_count += 1
#         person_ids.append(dataset.image_info[image_id]["id"])
#         img_ids.append(image_id)
#         origin = dataset.image_info[image_id]["path"]
#         move = origin.replace(data_type + "2017", data_type + "_person")
#         if(not os.path.exists(move[:-16])): os.mkdir(move[:-16])
#         shutil.copy(origin, move)

# print(f'Total person count - {person_count} images')

# numpy_data = np.array([np.array(person_ids)]).transpose()
# df = pd.DataFrame(data=numpy_data, columns=["Person_ID"])
# df.to_csv(COCO_DIR + "/annotations/1person_IDs.csv")

#--- 1-Person, 30-area Selection 
person_area_count = 0
person_ids = []
img_ids = []
for i in tqdm(range(len(dataset.image_ids))):
    image_id = dataset.image_ids[i]
    mask, class_ids = dataset.load_mask(image_id)
    if(Counter(class_ids)[1]==1): 
        total_area = dataset.image_info[image_id]['height'] * dataset.image_info[image_id]['width']
        area = int((sum(sum(mask * 1))[0] / total_area * 100))
        if(area > 30):
            person_area_count += 1
            person_ids.append(dataset.image_info[image_id]["id"])
            img_ids.append(image_id)
            origin = dataset.image_info[image_id]["path"]
            move = origin.replace(data_type + "2017", data_type + "_person_area")
            if(not os.path.exists(move[:-16])): os.mkdir(move[:-16])
            shutil.copy(origin, move)

print(f'Total person-area count - {person_area_count} images')

numpy_data = np.array([np.array(person_ids)]).transpose()
df = pd.DataFrame(data=numpy_data, columns=["Person_ID"])
df.to_csv(COCO_DIR + "/annotations/1person_30area_IDs.csv")

#--- Save outputs _masks        
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from collections import Counter
out_dir = os.path.join(ROOT_DIR, "output/mask/" + str(datetime.now())[:-10].replace(":", "_"))
if not os.path.exists(out_dir): os.makedirs(out_dir)

for i in tqdm(range(len(img_ids))):
    image_id = img_ids[i]
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    mask = mask[:, :, 0] * 1
    mask =np.moveaxis(np.stack([mask, mask, mask]), 0, 2)
    masked_image = image * mask
    name = dataset.image_info[image_id]["path"][-16:-4]
    saved = cv2.imwrite(f'{out_dir}/{name}_person.jpg', masked_image)

print("Finished..")
