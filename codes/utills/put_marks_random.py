# python put_marks_random.py
import cv2 as cv
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from resizeimage import resizeimage
import random

# Root directory of the project
ROOT_DIR = "../../"
IMAGE_DIR = os.path.join(ROOT_DIR, "data/random_images/")
COCO_DIR = "/media/farhat/Farhat_SSD/MarkNET" + "/data/coco/"

total_images = os.listdir(IMAGE_DIR)
total_images = [IMAGE_DIR +img for img in total_images if ".jpg" in img ]

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

prepapred = np.load(COCO_DIR + "train/" + 'marks_distribution_' + "train" + '.npz')
details = list(prepapred['details'])

indices = []
for detail in details:
    ind = []
    for i in range(detail[2]): ind.append(detail[1])
    if(detail[3]) : ind.append(detail[4])
    indices.append(ind)

paths = []
count = 0
bbox = []
point = []
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
    mark_index = random.choice(indices)
    print(f'{i+1} - Number of marks placed in image - {len(mark_index)}')

    cv.namedWindow('draw')
    cv.setMouseCallback('draw', mouse)
    
    path = total_images[i]

    bg = cv.imread(path)
    bgCopy = bg.copy()
    bg_pi = Image.open(path).copy()

    while True:
        cv.imshow('draw', bg)
        k = cv.waitKey(1)

        if (k == 27 & 0xFF) or ((len(point)//2)==len(mark_index)):
            point = np.array(point).reshape((-1, 2))
            paths.append(path)
            count += 1
            print("done - " + str(count))
            cv.destroyAllWindows()
            # Put the marks on selected points
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
            bg_pi.save(path.replace("/random_images/", "/random_images/mark_added/"))
            
            point = []
            break
    
        elif k == 32:
            print("Exit.. Total - " + str(count))
            cv.destroyAllWindows()
            break

    if k == 32:
        break
