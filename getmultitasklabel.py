import os
import numpy as np
import json
from glob import glob
import cv2
import shutil
import yaml
from tqdm import tqdm

# image_dir = r'G:\backup\project\yolov5-multitask\data\widerface_head_body_pose_landmark_yolo\train\images'
label_dir = r'G:\hp_tracking_proj\widerface_head_body_pose_lmk\train\labels'
label_image_des_dir = r'D:\Users\yl3146\Desktop\lastt'
# externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
# files = list()
# for extern in externs:
#     files.extend(glob(image_dir + "\\*." + extern))
#
# image_name = [i.replace("\\", "/").split("/")[-1].split('.')[0] for i in files]
# assert len(files) == len(image_name)


special_file = ['0--Parade', '1--Handshaking', '10--People_Marching', '11--Meeting', '12--Group', '13--Interview',
                '14--Traffic', '15--Stock_Market', '16--Award_Ceremony', '17--Ceremony', '18--Concerts', '19--Couple',
                '2--Demonstration', '20--Family_Group', '21--Festival', '22--Picnic', '23--Shoppers',
                '24--Soldier_Firing', '25--Soldier_Patrol', '26--Soldier_Drilling', '27--Spa', '28--Sports_Fan',
                '29--Students_Schoolkids', '3--Riot', '30--Surgeons', '31--Waiter_Waitress', '32--Worker_Laborer',
                '33--Running', '34--Baseball', '35--Basketball', '36--Football', '37--Soccer', '38--Tennis',
                '39--Ice_Skating', '4--Dancing', '40--Gymnastics', '41--Swimming', '42--Car_Racing', '43--Row_Boat',
                '44--Aerobics', '45--Balloonist', '46--Jockey', '47--Matador_Bullfighter',
                '48--Parachutist_Paratrooper', '49--Greeting', '5--Car_Accident', '50--Celebration_Or_Party',
                '51--Dresses', '52--Photographers', '53--Raid', '54--Rescue', '55--Sports_Coach_Trainer', '56--Voter',
                '57--Angler', '58--Hockey', '59--people--driving--car', '6--Funeral', '61--Street_Battle',
                '7--Cheering', '8--Election_Campain', '9--Press_Conference']

files = glob(label_dir + "\\*.txt")

out_file = open('%s/%s.txt' % (label_image_des_dir, 'hpf_lable'), 'w')
out_file_name = ''
for filename in tqdm(files):
    txt_name = filename.replace("\\", "/").split("/")[-1]
    tmp = txt_name.split('_')
    file_num, file_name = tmp[0], tmp[1]

    for name in special_file:
        name_num = name.split('--')[0]
        if name_num == file_num:
            out_file_name = name
            break
    out_file_name = out_file_name + "/"
    image_name = txt_name.split('.')[0] + '.jpg'
    save_file_name = "# " + out_file_name + image_name + "\n"
    label_txt = open(filename, 'r')
    lines = label_txt.readlines()
    out_file.write(save_file_name)
    for line in lines:
        out_file.write(line)
