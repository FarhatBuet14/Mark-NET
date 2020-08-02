# python BAG_cls_det2.py
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random, sys

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt

sys.path.append(os.path.join("../content/Mask_RCNN/samples/coco/"))  
import coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from detectron2.data.datasets import register_coco_instances

data_dir = "/media/farhat/Farhat_SSD/MarkNET/data/coco/"
train_json = data_dir + "train/annotations/instances_train2017.json"
val_json = data_dir + "val/annotations/instances_val2017.json"
train_img_dir = data_dir + "train/train_person_area_face"
val_img_dir = data_dir + "val/val_person_area_face"

register_coco_instances("my_dataset_train", {}, train_json, train_img_dir)
register_coco_instances("my_dataset_val", {}, val_json, val_img_dir)

from detectron2.structures import BoxMode
import pandas as pd

def get_mark_dicts(data_type):
    
    data = pd.read_csv(data_dir + data_type + "/annotations/1person_IDs.csv")
    person_ids = list(data.iloc[:, 1].values)
    # Load dataset
    dataset = coco.CocoDataset()
    dataset.load_coco(data_dir+data_type, subset= data_type, year="2017", class_ids = [1], image_ids=person_ids)

    # Must call before using the dataset
    dataset.prepare()
    imgs_anns = dataset.image_info

    dataset_dicts = []
    
    for idx, v in enumerate(imgs_anns):
        record = {}
    
        record["file_name"] = v["path"]
        record["image_id"] = idx
        record["height"] = v["height"]
        record["width"] = v["width"]

        annos = v["annotations"]
        objs = []
        for anno in annos:
            obj = {
                "area": anno["area"],
                "iscrowd": anno["iscrowd"],
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                # "segmentation": anno["segmentation"],
                "category_id": anno["category_id"],
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("mark_" + d, lambda d=d: get_mark_dicts(d))
    MetadataCatalog.get("mark_" + d).set(thing_classes=['BG', 'person'])
mark_metadata = MetadataCatalog.get("mark_train")

dataset_dicts = get_mark_dicts("train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=mark_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("mark_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (mark)
cfg.num_gpus = 1
# cfg.MODEL.WEIGHTS = os.path.join("../models/pretrained/detectron2/model_final_280758.pkl")

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=True)
# trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0034999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
cfg.DATASETS.TEST = ("mark_val", )
predictor = DefaultPredictor(cfg)

im = cv2.imread("./input.jpg")
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(out.get_image()[:, :, ::-1])
plt.show()

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_mark_dicts("val")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=mark_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("mark_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "mark_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

