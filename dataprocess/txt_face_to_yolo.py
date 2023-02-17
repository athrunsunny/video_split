"""
实现两个功能
1、转换face和lmk标注的数据格式
2、将hp和fl的标注合并
"""

import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm


# txt_labels_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\eee'
# img_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\img'
#
# des_label_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\labels'
# des_image_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\images'

def change_raw_label_format(origin_label, image_path, des_label_path, des_image_path):
    # 用途：1、取image_path下的所有图片；
    # 2、根据图片名取到相应的label的txt文件；
    # 3、txt文件的标注格式为x1y1x2y2的形式，将其转换为xywh，再将其转换为yolo的cxcywh/wh的形式

    # 输出的格式 [2,cx,cy,w,h, x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
    files = glob(origin_label + "\\*.txt")
    for txt_file in files:
        labels = open(txt_file, 'r')
        lines = labels.readlines()

        for line in lines:
            line = line.split(' ')
            padding = [-1.0, -1.0, -1.0]
            label = [float(x) for x in line if x != '']
            print(label)

    files = glob(image_path + "\\*.jpg")
    for path in tqdm(files):
        image_name = path.replace('\\', '/').split('/')[-1]
        # 默认是jpg格式的图片
        real_image_name = image_name[:-4]
        image_n = image_name.split('.')[0]
        img = cv2.imread(path)
        height, width, _ = img.shape

        label_path = os.path.join(origin_label, image_n + '.txt')
        labels = open(label_path, 'r')
        lines = labels.readlines()
        face_labels = list()

        for line in lines:
            annotation = np.zeros((1, 14))
            line = line.split(' ')
            padding = [-1.0, -1.0, -1.0]
            label = [float(x) for x in line if x != '']

            # bbox
            # 数据格式为x1y1x2y2
            label[0] = max(0, label[0])
            label[1] = max(0, label[1])
            label[2] = label[2] - label[0]
            label[3] = label[3] - label[1]

            label[2] = min(width - 1, label[2])
            label[3] = min(height - 1, label[3])
            annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
            annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
            annotation[0, 2] = label[2] / width  # w
            annotation[0, 3] = label[3] / height  # h

            # landmarks
            annotation[0, 4] = label[4] / width  # l0_x
            annotation[0, 5] = label[5] / height  # l0_y
            annotation[0, 6] = label[6] / width  # l1_x
            annotation[0, 7] = label[7] / height  # l1_y
            annotation[0, 8] = label[8] / width  # l2_x
            annotation[0, 9] = label[9] / height  # l2_y
            annotation[0, 10] = label[10] / width  # l3_x
            annotation[0, 11] = label[11] / height  # l3_y
            annotation[0, 12] = label[12] / width  # l4_x
            annotation[0, 13] = label[13] / height  # l4_y

            annotation = annotation.tolist()
            annotation = [2] + annotation[0] + padding
            face_labels.append(annotation)

        if real_image_name != image_n:
            image_n = real_image_name
        out_file = open('%s/%s.txt' % (des_label_path, image_n), 'w')
        for label in face_labels:
            if len(label) != 0:
                out_file.write(" ".join([str(p) for p in label]) + '\n')
        shutil.copy(path, des_image_path)


# 解开替换原始标注以及图片的路径，并修改目标路径
# if __name__ == '__main__':
#     txt_labels_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\eee'
#     img_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\img'
#
#     des_label_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\labels'
#     des_image_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\labels\images'
#
#     change_raw_label_format(txt_labels_path, img_path, des_label_path, des_image_path)


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_ori_head_person(hp_label_dir, image_name):
    # 这里的数据由labelme标注并由toyolo.py生成
    head_label = list()
    # for txt_name in image_name:
    labeltxt = os.path.join(hp_label_dir, image_name + '.txt')
    if not os.path.isfile(labeltxt):
        return [], 0
    head_label_txt = open(labeltxt, 'r')
    lines = head_label_txt.readlines()

    for line in lines:
        padding = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        line = line.split(' ')
        label = [float(x) for x in line if x != '']
        head_label.append(label + padding)

    read_time = 0
    if head_label:
        read_time = 1
    return head_label, read_time


def read_ori_face_lmk(fl_labels_path, image_name):
    face_label = list()

    # for txt_name in image_name:
    labeltxt = os.path.join(fl_labels_path, image_name + '.txt')
    if not os.path.isfile(labeltxt):
        return [], 0
    face_label_txt = open(labeltxt, 'r')
    lines = face_label_txt.readlines()

    for line in lines:
        line = line.split(' ')
        label = [float(x) for x in line if x != '']
        face_label.append(label)
    read_time = 0
    if face_label:
        read_time = 1
    return face_label, read_time


def cat_person_head_face_lmk_labels(image_dir, hp_label_dir, fl_labels_dir, des_image_dir, des_label_dir):
    """
    :param image_dir: 原始的图片数据路径
    :param hp_label_dir: 带有person和head标注的数据路径
    :param fl_labels_dir: 带有face和lmk标注的数据路径
    :param des_image_dir: 图片存放路径
    :param des_label_dir: 标签存放路径
    :return:
    """
    files = glob(image_dir + "\\*.jpg")
    create_path(des_image_dir)
    create_path(des_label_dir)
    for idx, img in enumerate(files):
        image_name = img.replace('\\', '/').split('/')[-1]
        real_image_name = image_name[:-4]

        idx_str = str(idx)
        change_img_name = "62_office_station_" + (4 - len(idx_str)) * '0' + str(idx)

        image_n = f'{des_image_dir}/{change_img_name}.jpg'
        out_file = open('%s/%s.txt' % (des_label_dir, change_img_name), 'w')

        hp_label, h_time = read_ori_head_person(hp_label_dir, real_image_name)
        face_label, f_time = read_ori_face_lmk(fl_labels_dir, real_image_name)
        last_label = hp_label + face_label

        for label in last_label:
            if len(label) != 0:
                out_file.write(" ".join([str(p) for p in label]) + '\n')
        # 统一图像格式成jpg
        image = cv2.imread(img)
        cv2.imwrite(image_n, image)


# 解开后可以合并标注，并将图片和标注移动到指定文件夹中
# if __name__ == '__main__':
#     image_dir = r'G:\hp_tracking_proj\add_new_data\face_lmk\train\images'
#     hp_labels_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\train\labels'
#     des_train_image_des_dir = r'G:\hp_tracking_proj\add_new_data\face_lmk\gt\images'
#     des_label_image_des_dir = r'G:\hp_tracking_proj\add_new_data\face_lmk\gt\labels'
#     fl_labels_path = r'G:\hp_tracking_proj\add_new_data\face_lmk\fl_gt_yolotype\labels'
#     cat_person_head_face_lmk_labels(image_dir, hp_labels_path, fl_labels_path, des_train_image_des_dir,
#                                     des_label_image_des_dir)
