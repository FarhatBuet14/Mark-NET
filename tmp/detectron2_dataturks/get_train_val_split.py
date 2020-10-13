import os
import numpy as np
import pandas as pd

files = os.listdir()

files = [file if (".jpeg" in file) else None for file in files]
files = [i for i in files if i] 

print(f'Test files - {len(files)}')
df = pd.DataFrame(files)
df.to_csv("Test_Names.csv")

all_files = os.listdir("/media/farhat/Farhat_reserved/Research/GitHub/FaceMarkNET/data/Dataturks/faces/")
all_files = [file if (".jpeg" in file) else None for file in all_files]
all_files = [i for i in all_files if i] 

train_files = [file if (file not in files) else None for file in all_files]
train_files = [i for i in train_files if i] 

print(f'Training files - {len(train_files)}')
df = pd.DataFrame(train_files)
df.to_csv("Train_Names.csv")

print("Finished")