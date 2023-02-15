import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm


txt_labels_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\eee'
files = glob(txt_labels_path + "\\*.txt")
for txt_file in files:
    labels = open(txt_file, 'r')
    lines = labels.readlines()

    for line in lines:
        line = line.split(' ')
        padding = [-1.0, -1.0, -1.0]
        label = [float(x) for x in line if x != '']
        print(label)
