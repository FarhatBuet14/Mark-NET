import sys
import random
import math
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# To find local version - https://github.com/matterport/Mask_RCNN
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))  
import coco
from pycocotools.coco import COCO


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# # Path Specified

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# Download from - https://github.com/matterport/Mask_RCNN/releases
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/pretrained/mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data/random_images")


# # Configuration


class InferenceConfig(coco.CocoConfig):
  #class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# # Model with pretrained weights

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)
# Load weights trained on MS-COCO
from keras.engine import saving
model.load_weights(COCO_MODEL_PATH, by_name=True)


# # COCO dataset samples
# # COCO Class names - total 80 classes

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# # Loading image and prediction

# Load a random image from the images folder
# file_name = random.choice(os.listdir(IMAGE_DIR))
file_name = "15.jpg"
image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

import matplotlib.pyplot as plt
plt.figure(figsize=(16,16))
plt.show(image)


# # Prediction

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
from collections import Counter
print(f'Total detected objects - {len(r["class_ids"])}')
print(f'Total unique objects - {len(np.unique(r["class_ids"]))}')
print("--------------------------------")
bla = [print(f'{class_names[id]} - {num}') for id,num in Counter(r['class_ids']).items()]


# # Showing Masks

import matplotlib.pyplot as plt
ids = np.where(r['class_ids']==class_names.index("person"))[0]
ids = ids if(len(ids)<6) else ids[:10]
plt.figure(figsize=(30,30))
columns = 2
for i, id in enumerate(ids):
    mask = r['masks'][:, :, id] * 1
    mask =np.moveaxis(np.stack([mask, mask, mask]), 0, 2)
    masked_image = image * mask
    plt.subplot(len(ids) / columns + 1, columns, i + 1)
    plt.show(masked_image)

from mrcnn.visualize import display_images
import mrcnn.model as modellib
display_images(np.transpose(r['masks'], [2, 0, 1]), cmap="Blues")


# Save outputs

import matplotlib.pyplot as plt
import cv2
import shutil
from datetime import datetime
out_dir = os.path.join(ROOT_DIR, "outputs/mask/" + str(datetime.now())[:-10].replace(":", "_"))
if not os.path.exists(out_dir): os.makedirs(out_dir)
else: 
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)
for i, id in enumerate(r['class_ids']):
    mask = r['masks'][:, :, i] * 1
    mask =np.moveaxis(np.stack([mask, mask, mask]), 0, 2)
    masked_image = image * mask
    bla = cv2.imwrite(f'{out_dir}/{i+1}_{class_names[id]}.jpg', masked_image)

