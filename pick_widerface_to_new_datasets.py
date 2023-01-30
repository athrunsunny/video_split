import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm
import random

image_file_dirs = r'G:\hp_tracking_proj\WIDER_train\images'

seed = 32
random.seed(seed)
np.random.seed(seed)


def get_val_files(image_file_dir, save_prob=0.18):
    file_name = list()
    for file in os.listdir(image_file_dir):
        file_path = os.path.join(image_file_dir, file)
        for image in os.listdir(file_path):
            prob = np.random.random()
            if prob <= save_prob:
                file_name.append(image)
    return file_name


files = get_val_files(image_file_dirs)

wait_process_path = r'G:\hp_tracking_proj\widerface_head_body_pose_lmk\train_images_labels'
images_path = os.path.join(wait_process_path, 'images')
labels_path = os.path.join(wait_process_path, 'labels')


def create_path(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    return path_dir


train_path = create_path(os.path.join(wait_process_path, 'train'))
val_path = create_path(os.path.join(wait_process_path, 'val'))

timages_path = create_path(os.path.join(train_path, 'images'))
tlabels_path = create_path(os.path.join(train_path, 'labels'))

vimages_path = create_path(os.path.join(val_path, 'images'))
vlabels_path = create_path(os.path.join(val_path, 'labels'))

files = [i.replace("\\", "/").split("/")[-1].split(".")[0] for i in files]

externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
images = list()
for extern in externs:
    images.extend(glob(images_path + "\\*." + extern))

labels = glob(labels_path + "\\*.txt")


for image,label in tqdm(zip(images,labels)):
    image_name = image.replace("\\", "/").split("/")[-1].split(".")[0]
    if image_name in files:
        shutil.copy(image, vimages_path)
        shutil.copy(label, vlabels_path)
    else:
        shutil.copy(image, timages_path)
        shutil.copy(label, tlabels_path)
print('done!')

