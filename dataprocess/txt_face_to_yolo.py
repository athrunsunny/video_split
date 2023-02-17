import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm


txt_labels_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\eee'
img_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\img'

des_label_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\labels'
des_image_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\images'
# files = glob(txt_labels_path + "\\*.txt")
# for txt_file in files:
#     labels = open(txt_file, 'r')
#     lines = labels.readlines()
#
#     for line in lines:
#         line = line.split(' ')
#         padding = [-1.0, -1.0, -1.0]
#         label = [float(x) for x in line if x != '']
#         print(label)

# files = glob(img_path + "\\*.jpg")
# for path in files:
#     image_name = path.replace('\\', '/').split('/')[-1]
#     real_image_name = image_name[:-4]
#     image_n = image_name.split('.')[0]
#     img = cv2.imread(path)
#     height, width, _ = img.shape
#
#     label_path = os.path.join(txt_labels_path,image_n+'.txt')
#     labels = open(label_path,'r')
#     lines = labels.readlines()
#     face_labels = list()
#     print(path)
#     for line in lines:
#         annotation = np.zeros((1, 14))
#         line = line.split(' ')
#         padding = [-1.0, -1.0, -1.0]
#         label = [float(x) for x in line if x != '']
#         # print(label)
#         # bbox
#         label[0] = max(0, label[0])
#         label[1] = max(0, label[1])
#         label[2] = label[2] - label[0]
#         label[3] = label[3] - label[1]
#         # print(label)
#         label[2] = min(width - 1, label[2])
#         label[3] = min(height - 1, label[3])
#         annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
#         annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
#         annotation[0, 2] = label[2] / width  # w
#         annotation[0, 3] = label[3] / height  # h
#         # if (label[2] -label[0]) < 8 or (label[3] - label[1]) < 8:
#         #    img[int(label[1]):int(label[3]), int(label[0]):int(label[2])] = 127
#         #    continue
#         # landmarks
#         annotation[0, 4] = label[4] / width  # l0_x
#         annotation[0, 5] = label[5] / height  # l0_y
#         annotation[0, 6] = label[6] / width  # l1_x
#         annotation[0, 7] = label[7] / height  # l1_y
#         annotation[0, 8] = label[8] / width  # l2_x
#         annotation[0, 9] = label[9] / height  # l2_y
#         annotation[0, 10] = label[10] / width  # l3_x
#         annotation[0, 11] = label[11] / height  # l3_y
#         annotation[0, 12] = label[12] / width  # l4_x
#         annotation[0, 13] = label[13] / height  # l4_y
#         # print(annotation)
#         annotation = annotation.tolist()
#         annotation = [2] + annotation[0] + padding
#         # print(annotation)
#         face_labels.append(annotation)
#
#     if real_image_name != image_n:
#         image_n = real_image_name
#     out_file = open('%s/%s.txt' % (des_label_path, image_n), 'w')
#     for label in face_labels:
#         if len(label) != 0:
#             out_file.write(" ".join([str(p) for p in label]) + '\n')
#     shutil.copy(path, des_image_path)


train_image_des_dir = r'G:\hp_tracking_proj\add_new_data\face_lmk\train\images'
if not os.path.exists(train_image_des_dir):
    os.makedirs(train_image_des_dir)

label_image_des_dir = r'G:\hp_tracking_proj\add_new_data\face_lmk\train\labels'
if not os.path.exists(label_image_des_dir):
    os.makedirs(label_image_des_dir)

des_train_image_des_dir = r'G:\hp_tracking_proj\add_new_data\face_lmk\gt\images'
if not os.path.exists(des_train_image_des_dir):
    os.makedirs(des_train_image_des_dir)

des_label_image_des_dir = r'G:\hp_tracking_proj\add_new_data\face_lmk\gt\labels'
if not os.path.exists(des_label_image_des_dir):
    os.makedirs(des_label_image_des_dir)

fl_labels_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\fl_gt_yolotype\labels'


def read_ori_head_person(image_name):
    head_label = list()
    # for txt_name in image_name:
    labeltxt = os.path.join(label_image_des_dir, image_name + '.txt')
    if not os.path.isfile(labeltxt):
        return [],0
    head_label_txt = open(labeltxt, 'r')
    lines = head_label_txt.readlines()

    for line in lines:
        # if line[0] == '0.0' or line[0] == '0':
            # padding = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
            # 保持元数据的对齐
        padding = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        line = line.split(' ')
        label = [float(x) for x in line if x != '']
        # label[0] = int(label[0] + 1)
        head_label.append(label + padding)

    read_time = 0
    if head_label:
        read_time = 1
    return head_label, read_time


def read_oir_face_lmk(image_name):
    face_label = list()

    # for txt_name in image_name:
    labeltxt = os.path.join(fl_labels_path, image_name + '.txt')
    if not os.path.isfile(labeltxt):
        return [],0
    face_label_txt = open(labeltxt, 'r')
    lines = face_label_txt.readlines()

    for line in lines:
        # if line[0] == '0.0' or line[0] == '0':
            # line = line.split(' ')[:-2]
            # 保持元数据的对齐
            # padding = [-1.0, -1.0, -1.0]
        line = line.split(' ')
        label = [float(x) for x in line if x != '']
        # label[0] = int(label[0] + 2)
        # face_label.append(label + padding)
        face_label.append(label)
    read_time = 0
    if face_label:
        read_time = 1
    return face_label, read_time

files = glob(train_image_des_dir + "\\*.jpg")
for idx,img in enumerate(files):
    image_name = img.replace('\\', '/').split('/')[-1]
    real_image_name = image_name[:-4]

    idx_str = str(idx)
    change_img_name = "62_office_station_" + (4 - len(idx_str)) * '0' + str(idx)

    image_n = f'{des_train_image_des_dir}/{change_img_name}.jpg'
    out_file = open('%s/%s.txt' % (des_label_image_des_dir, change_img_name), 'w')

    hp_label, h_time = read_ori_head_person(real_image_name)
    face_label, f_time = read_oir_face_lmk(real_image_name)
    last_label = hp_label+face_label

    for label in last_label:
        if len(label) != 0:
            out_file.write(" ".join([str(p) for p in label]) + '\n')

    image = cv2.imread(img)
    cv2.imwrite(image_n,image)

