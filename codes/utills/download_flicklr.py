# python download_flicklr.py
import json
import urllib.request as ul
import json
import PIL.Image as Image
import cv2
import os
from tqdm import tqdm

data_dir = "../../data/flicklr/"

with open(data_dir + 'annotations/ffhq-dataset-v2.json') as f:
  data = json.load(f)

links = [img["image"]["file_url"] for img in [data[str(i)] for i in range(1,len(data))]]

paths = [img["image"]["file_path"][:-9] for img in [data[str(i)] for i in range(1,len(data))]]

names = [img["image"]["file_path"][-9:] for img in [data[str(i)] for i in range(1,len(data))]]

img_dir = data_dir + "images/"

print("Start downloading...")

for i in tqdm(range(len(data))):
  if(not os.path.exists(img_dir + paths[i])): os.mkdir(img_dir + paths[i])
  img = ul.urlopen(links[i])
  img = Image.open(img)
  img = img.convert('RGB')
  img.save(img_dir + paths[i] + names[i], "JPEG")

print("Finish downloading...")

