import cv2 as cv
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from resizeimage import resizeimage
import random

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
pi_marks = []
marks_name = []
for folder in os.listdir(mark_dir):
    m, pi_m, n = [], [], []
    for file in os.listdir(mark_dir + folder):
        m.append(cv.imread(os.path.join(mark_dir, folder, file), 0))
        pi_m.append(Image.open(os.path.join(mark_dir, folder, file)))
        n.append(file)
    marks.append(m)
    pi_marks.append(pi_m)
    marks_name.append(n)

total_images = [info["path"].replace("2017/", "_person_area_face_masks/") for info in dataset.image_info]
total_data = len(total_images)

# --- Splitting Marks
tot_mark = 5
img_per_mark = (total_data // tot_mark) + 1
details = []
for i in range(tot_mark):
    index_mark = i
    in_start = (i*img_per_mark)
    if(i==(tot_mark-1)): in_end = total_data
    else: in_end = (i*img_per_mark + img_per_mark)
    # print(f'Start - {in_start} End {in_end}')
    for j in range(in_start, in_end):
        m = [0, 1, 2, 3, 4]
        m.remove(index_mark)
        details.append((j,index_mark, random.randrange(1, 4), 
                random.choice([True, False]), random.choice(m)))

indices = []
for detail in details:
    ind = []
    for i in range(detail[2]): ind.append(detail[1])
    if(detail[3]) : ind.append(detail[4])
    indices.append(ind)


# from collections import Counter
# Counter([i[2] for i in details])
# Counter([i[3] for i in details])

# # ---- Load past data
# prepapred_data = np.load(COCO_DIR + 'marks_point_' + data_type + '.npz')
# points = list(prepapred_data['manual_points'])
# bbox = list(prepapred_data['bbox'])
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
bbox = []

for i in range(len(total_images)):

    def mouse(event,x,y,flags,params):
        global move_rectangle, BLUE, bg, bgCopy, point
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
    
    rows, cols = (69, 55) # fg.shape[:2]
    mark_index = indices[i]
    print(f'{i+1} - Number of marks placed in image - {len(mark_index)}')

    cv.namedWindow('draw')
    cv.setMouseCallback('draw', mouse)
    
    # info = dataset.image_info[i]
    # path = info["path"].replace("2017/", "_person_area_face_masks/")
    path = total_images[i]

    bg = cv.imread(path)
    bgCopy = bg.copy()
    bg_pi = Image.open(path).copy()

    while True:
        cv.imshow('draw', bg)
        k = cv.waitKey(1)

        if (k == 27 & 0xFF) or ((len(point)//2)==len(mark_index)):
            point = np.array(point).reshape((-1, 2))
            points.append(point)
            paths.append(path)
            count += 1
            print("done - " + str(count))
            cv.destroyAllWindows()
            # Put the marks on selected points
            # fg = marks[index_mark]
            # index_list = [i for i in range(0, len(pi_marks))]
            # fg_pi = pi_marks[index_mark].copy()
            box = []
            for i, m in enumerate(mark_index):
                fg_pi = pi_marks[m][random.choice([0, 1, 2, 3])].copy()
                re_size = random.choice([1, 2, 3, 4])
                fg_pi = resizeimage.resize_thumbnail(fg_pi, [fg_pi.size[0] // re_size, fg_pi.size[1] // re_size])
                cox, coy = int(point[i][0]), int(point[i][1])
                box.append([cox, coy, int(fg_pi.size[1]), int(fg_pi.size[0])]) # x, y, h, w
                bg_pi.paste(fg_pi, (cox, coy), fg_pi)
            
            bbox.append(box)
            bg_pi.show()
            bg_pi.save(path.replace("_masks", "_masks_marks"))
            
            point = []
            break
    
        elif k == 32:
            print("Exit.. Total - " + str(count))
            cv.destroyAllWindows()
            break

    if k == 32:
        break

# ---- Save data
np.savez(COCO_DIR + 'marks_point_' + data_type + '.npz',
        manual_points = points, bbox = bbox, 
        paths = paths)
