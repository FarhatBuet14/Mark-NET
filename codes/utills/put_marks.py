import cv2 as cv
import os
import sys
import numpy as np
import pandas as pd

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils

from coco import coco
config = coco.CocoConfig()
data_type = "val" #val #test
COCO_DIR = "/media/farhat/Farhat_SSD/MarkNET" + "/data/coco/" + data_type


data = pd.read_csv(COCO_DIR + "/annotations/1person_30area_face_IDs.csv")
person_ids = list(data.iloc[:, 1].values)

# Load dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, data_type, class_ids = [1], image_ids=person_ids)

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


mark_dir = ROOT_DIR + "/data/marks/"
marks = []
for file in os.listdir(mark_dir):
    marks.append(cv.imread(mark_dir + file, 0))

index_mark = 0

total_images = [info["path"].replace("2017/", "_person_area_face_masks/") for info in dataset.image_info]
total_data = len(total_images)

# # ---- Load past data
# prepapred_data = np.load(COCO_DIR + 'marks_point_' + data_type + '.npz')
# points = list(prepapred_data['manual_points'])
# paths = list(prepapred_data['paths'])
# count = len(points)
# for name in paths:
#    if(str(name) in total_images):
#       total_images.remove(str(name)) 
# if(total_data - len(total_images) == count):
#    print("Successfully added the past data")

point = []
paths = []
points = []
count = 0

for i in range(len(total_images)):

    def mouse(event,x,y,flags,params):
        global move_rectangle, BLUE, fg, bg, bgCopy, point
        #draw rectangle where x,y is rectangle center
        if event == cv.EVENT_LBUTTONDOWN:
            move_rectangle = True

        elif event == cv.EVENT_MOUSEMOVE:
            bg = bgCopy.copy() #!! your image is reinitialized with initial one
            if move_rectangle:
                cv.rectangle(bg,(x-int(0.5*cols),y-int(0.5*rows)),
                (x+int(0.5*cols),y+int(0.5*rows)),BLUE, -1)

        elif event == cv.EVENT_LBUTTONUP:
            move_rectangle = False
            cv.rectangle(bg,(x-int(0.5*cols),y-int(0.5*rows)),
            (x+int(0.5*cols),y+int(0.5*rows)),BLUE, -1)
        
        elif event == cv.EVENT_RBUTTONDOWN:
            print(str(x) + " , " + str(y))
            point.append(x)
            point.append(y)
    
    move_rectangle = False
    BLUE = [255,0,0]

    fg = marks[index_mark]
    
    
    rows, cols = fg.shape[:2]

    cv.namedWindow('draw')
    cv.setMouseCallback('draw', mouse)
    
    # info = dataset.image_info[i]
    # path = info["path"].replace("2017/", "_person_area_face_masks/")
    path = total_images[i]

    bg = cv.imread(path)
    bgCopy = bg.copy()

    while True:
        cv.imshow('draw', bg)
        k = cv.waitKey(1)

        if k == 27 & 0xFF:
            points.append(point)
            paths.append(path)
            point = []
            count += 1
            print("done - " + str(count))
            cv.destroyAllWindows()
            break
    
        elif k == 32:
            print("Exit.. Total - " + str(count))
            cv.destroyAllWindows()
            break
    if k == 32:
        break

points = np.array(points)

# ---- Save data
np.savez(COCO_DIR + 'marks_point_' + data_type + '.npz',
        manual_points = points,
        paths = paths)































