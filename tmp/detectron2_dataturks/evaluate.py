# python evaluate.py
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import ntpath
import numpy as np
import cv2
import random
import itertools
import pandas as pd
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

import torch

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

data_dir = "./data/Dataturks/"
df = pd.read_csv(data_dir + 'annotations.csv')
IMAGES_PATH = data_dir + 'faces'

train_files = pd.read_csv(data_dir + 'Train_Names.csv')
train_files = train_files.iloc[:, 1].to_list()
train_df = df[df.file_name.isin(train_files)]
test_df = df[~df.file_name.isin(train_files)]
classes = df.class_name.unique().tolist()

def create_dataset_dicts(df, classes):
    dataset_dicts = []
    for image_id, img_name in enumerate(df.file_name.unique()):
        record = {}

        image_df = df[df.file_name == img_name]

        file_path = f'{IMAGES_PATH}/{img_name}'
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

for d in ["train", "val"]:
    DatasetCatalog.register("faces_" + d, lambda d=d: create_dataset_dicts(train_df if d == "train" else test_df, classes))
    MetadataCatalog.get("faces_" + d).set(thing_classes=classes, stuff_classes = classes)

statement_metadata = MetadataCatalog.get("faces_train")

# Check data annotation
# dataset_dicts = create_dataset_dicts(train_df, classes)
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=statement_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow("", vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("faces_train",)
cfg.DATASETS.TEST = ("faces_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.CUDNN_BENCHMARK = True
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)

from datetime import datetime
output_dir = "./output/"
evaluator = COCOEvaluator("faces_val", cfg, False, output_dir)
val_loader = build_detection_test_loader(cfg, "faces_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

time = str(datetime.now()).replace(":", "_")[:-7]
output_face_dir= output_dir + time + "/Annotated_results"
os.makedirs(output_face_dir, exist_ok=True)
test_image_paths = test_df.file_name.unique()

for clothing_image in test_image_paths:
    file_path = f'{IMAGES_PATH}/{clothing_image}'
    im = cv2.imread(file_path)
    
    outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        metadata=statement_metadata,
        scale=1.,
        instance_mode=ColorMode.IMAGE
    )
    instances = outputs["instances"].to("cpu")
    # instances.remove('pred_masks')
    # v = v.draw_instance_predictions(instances)
    mask = instances.pred_masks
    mask = torch.tensor(np.array([m==False for m in mask.numpy()]))
    v = v.draw_sem_seg(mask[0])

    result = v.get_image()[:, :, ::-1]
    file_name = ntpath.basename(clothing_image)
    write_res = cv2.imwrite(f'{output_face_dir}/{file_name}', result)
