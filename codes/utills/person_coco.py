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
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))  
import coco
config = coco.CocoConfig()
COCO_DIR = "/media/farhat/Farhat_SSD/MarkNET" + "/data/coco/train"

# data = pd.read_csv(COCO_DIR + "/annotations/person_IDs.csv")
# person_ids = list(data.iloc[:, 1].values)

# Load dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, subset= "train", year="2017") # , class_ids = [1], image_ids=person_ids

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))

person_count = 0
person_ids = []
for i in tqdm(range(len(dataset.image_ids))):
    image_id = dataset.image_ids[i]
    _, class_ids = dataset.load_mask(image_id)
    if(Counter(class_ids)[1]==1): 
        person_count += 1
        person_ids.append(dataset.image_info[image_id]["id"])
        # origin = dataset.image_info[image_id]["path"]
        # shutil.copy(origin, origin.replace("2017", "_person"))

print(f'Total person count - {person_count} images')

numpy_data = np.array([np.array(person_ids)]).transpose()
df = pd.DataFrame(data=numpy_data, columns=["Person_ID"])
df.to_csv(COCO_DIR + "/annotations/person_IDs.csv")

print("Finished..")
