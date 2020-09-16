# python mark_trainer_det2.py

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import glob
import os
import ntpath
import numpy as np
import cv2
import random
import itertools
import pandas as pd
from tqdm import tqdm
import urllib
import json
import PIL.Image as Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc


sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#########################################    Start Codes    #########################################

# --- Load Annotations
data_type = "train" #val #test
COCO_DIR = "/media/farhat/Farhat_SSD/MarkNET" + "/data/coco/" + data_type

prepapred_data = np.load(COCO_DIR + '/marks_point_' + data_type + '.npz', allow_pickle=True)
bbox = list(prepapred_data['bbox'])
paths = list(prepapred_data['paths'])
# class_name = [['mark' for j in range(len(bbox[i]))] for i in range(len(paths))]

# --- Test Annotations
# def annotate_image(pth, marks):
#     img = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)
#     for mark in marks:
#         x, y, h, w = mark
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     return img

# aList = [i for i in range(len(paths))]
# for i in random.sample(aList, 10):
#     path = paths[i].replace("/train_person_area_face_masks/", '/train_person_area_face_masks_marks/')
#     img = annotate_image(path, bbox[i])
#     cv2.imshow('ImageWindow', img)
#     cv2.waitKey()

# --- Save Annotations in CSV format
# paths = np.array(paths).transpose()
# bbox = np.array(bbox).transpose()


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
        data["x_max"] = int(round(x+w))
        data["y_max"] = int(round(y+h))
        data['class_name'] = 'mark'
        dataset.append(data)

df = pd.DataFrame(dataset)
df.to_csv(COCO_DIR + "/marks_annotations.csv", header=True, index=None)

# --- Train-Test Splitting
unique_files = df.file_name.unique()
train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.95), replace=False))
train_df = df[df.file_name.isin(train_files)]
test_df = df[~df.file_name.isin(train_files)]

classes = df.class_name.unique().tolist()

# --- Dataset Dictionary for Training
def create_dataset_dicts(df, classes):
    dataset_dicts = []
    for image_id, img_name in enumerate(df.file_name.unique()):
        record = {}
        image_df = df[df.file_name == img_name]
        file_path = img_name
        record["file_name"] = file_path
        record["image_id"] = image_id
        record["height"] = int(image_df.iloc[0].height)
        record["width"] = int(image_df.iloc[0].width)
        objs = []
        for _, row in image_df.iterrows():
            xmin = int(row.x_min)
            ymin = int(row.y_min)
            xmax = int(row.x_max)
            ymax = int(row.y_max)
            poly = [
                (xmin, ymin), (xmax, ymin),
                (xmax, ymax), (xmin, ymax)
            ]
            poly = list(itertools.chain.from_iterable(poly))
            obj = {
            "bbox": [xmin, ymin, xmax, ymax],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": classes.index(row.class_name),
            "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# --- Assign the Dictionary
for d in ["train", "val"]:
    DatasetCatalog.register("mark_" + d, lambda d=d: create_dataset_dicts(train_df if d == "train" else test_df, classes))
    MetadataCatalog.get("mark_" + d).set(thing_classes=classes)
statement_metadata = MetadataCatalog.get("mark_train")


# --- Set our own Trainer (add Evaluator for the test set )
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# --- Set the Configs
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("mark_train",)
cfg.DATASETS.TEST = ("mark_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.TEST.EVAL_PERIOD = 500

cfg.num_gpus = 1

# --- Start Training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("Finished Training...")
