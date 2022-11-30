import os
import argparse
import json
import cv2
import torch
import random
import string
from cvflow.utils import get_logger, Config
from cvflow.module.builder import build_modules
from cvflow.visualizer.builder import build_visualizers

ROOT_DIR = os.getcwd()


class LoadUSBcam:
    """
    读取摄像头数据
    """
    INFO = ['fps', 'fw', 'fh', 'bs']

    def __init__(self, pipe='0', **options):
        switch = options.pop('switch', True)  # False为默认相机输入大小
        frameWidth = options.pop('frameWidth', 1280)
        frameHeight = options.pop('frameHeight', 720)
        flip = options.pop('flip', False)
        bufferSize = options.pop('bufferSize', 10)
        running = options.pop('running', False)

        self.infoDict = dict()
        self.running = running
        if pipe.isnumeric():
            pipe = eval(pipe)
        self.pipe = pipe
        self.flip = flip

        self.cap = cv2.VideoCapture(self.pipe)
        if switch:
            self.frameWidth = frameWidth
            self.frameHeight = frameHeight
        else:
            self.frameWidth = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frameHeight = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.bufferSize = bufferSize
        self.setprocess()
        self.processeDict()

    def processeDict(self):
        self.infoDict['fps'] = self.fps
        self.infoDict['fw'] = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.infoDict['fh'] = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.infoDict['bs'] = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)

    def setprocess(self):
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.bufferSize)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))

    def __iter__(self):
        self.count = -1
        return self

    def __len__(self):
        return 0

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # 读取视频帧
        try:
            if self.pipe == 0:
                ret_val, frame = self.cap.read()
                if self.flip:
                    frame = cv2.flip(frame, 1)
            else:
                self.cap.grab()
                ret_val, frame = self.cap.retrieve()
                if self.flip:
                    frame = cv2.flip(frame, 1)
        except:
            raise StopIteration
        return frame

    def __getitem__(self, key):
        return self.infoDict[key]

    def getFrameCount(self):
        if isinstance(self.pipe, str):
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return int(self.count)

    def getFrameSize(self):
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return size

    def getFps(self):
        fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        return fps

    def getTime(self):
        if self.getFrameCount() == 0 or self.getFps() == 0:
            return 0
        videotime = round(self.getFrameCount() / self.getFps())
        return videotime

    def set(self, value):
        self.pipe = value
        self.cap = cv2.VideoCapture(self.pipe)
        self.setprocess()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


def check_config_file(config_file_path, video_file_path, input_file_path):
    """Check the pipeline of configure file. Make sure the module input requirement is met.
    Args:
        config_file_path (str): The path of the configuration file.
        video_file_path (str): The path of the video file.
        input_file_path (str): The path of the annotated file.
    Returns:
        True/False (bool): Whether the check is passed.
    """
    # read initial input attributes
    attributes = []
    if video_file_path:
        attributes.append("video")
    if input_file_path:
        input_attributes = json.loads(open(input_file_path).readline())
        for input_attr in input_attributes["attributes"]:
            attributes.append(input_attr)
    logger.info("initial input attributes: {}".format(attributes))

    # read config file and check each module
    cfg = Config.fromfile(config_file_path)
    for module in cfg.modules:
        for required_input in module.input:
            if required_input not in attributes:
                logger.error(
                    "required input attribute [%s] is not provided for module [%s]."
                    % (required_input, module["name"]))
                return False
        for produced_output in module.output:
            if produced_output not in attributes:
                attributes.append(produced_output)
    logger.info(
        "configuration check passed. output attributes: {}".format(attributes))
    return True


def parse_argument():
    parser = argparse.ArgumentParser(
        description="The video/image pipeline program.")
    parser.add_argument(
        "--config", default='configs/hand_gesture.py', help="configuration file path")
    parser.add_argument(
        "--video", default=r'F:\new_project1\test\L.mp4',
        help="input video path,"
             "or image file path,"
             "or image sequence path,"
             "represented like example/%04d.png",
    )
    parser.add_argument("--out_video", help="output video path")
    parser.add_argument("--input", help="the input annotation file path")
    parser.add_argument("--output", help="the output annotation file path")
    parser.add_argument("--video_data", default=ROOT_DIR, help="output video file path")
    parser.add_argument("--w", default=1920, type=int, help="video frame width")
    parser.add_argument("--h", default=1080, type=int, help="video frame height")
    parser.add_argument("--flip", default=False, help="video flip")
    parser.add_argument(
        "--log_level",
        choices=["debug", "info", "warn", "error"],
        help="the log level",
        default="info")
    parser.add_argument(
        "--gpus", default="-1", type=str, help="gpu ids, e.g. '0,1,2'")
    parser.add_argument(
        "--show", default=True, help="display the video", action="store_true")

    args = parser.parse_args()
    return args


def gather_attributes(input_file, module_dict):
    attributes = []
    if args.input:
        f_inputs = open(args.input, 'r')
        for attribute in json.loads(f_inputs.readline())["attributes"]:
            if attribute not in attributes:
                attributes.append(attribute)
    for module in module_dict:
        for attribute in module["output"]:
            if attribute not in attributes:
                attributes.append(attribute)
    return attributes


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def add_frame_id(itm_list, frameid):
    for pos in itm_list['relpos']:
        if check_label(itm_list[f'hand_class_{pos}']):
            if f'hand_{pos}_frameid' not in itm_list:
                itm_list[f'hand_{pos}_frameid'] = [frameid]
            else:
                itm_list[f'hand_{pos}_frameid'].append(frameid)
    return itm_list


def check_label(label_class):
    if label_class in ['L', 'P', 'F']:
        return True
    return False


def padding_zero(nc_dict):
    for item in nc_dict:
        if 'left' not in item['relpos']:
            item['hand_bbox_left'] = [0 for _ in range(5)]
        if 'right' not in item['relpos']:
            item['hand_bbox_right'] = [0 for _ in range(5)]
        item['relpos'] = ['left', 'right']
    return nc_dict


def limit_hand_area(personpre, personnow):
    for pospre in personpre['relpos']:
        for posaft in personnow['relpos']:
            if posaft == pospre:
                hand_bbox_pre = personpre[f'hand_bbox_{pospre}']
                hand_bbox_aft = personnow[f'hand_bbox_{posaft}']
                hand_iou = box_iou(torch.tensor([hand_bbox_pre[:4]]),
                                   torch.tensor([hand_bbox_aft[:4]]))
                if hand_iou > 0.90:
                    # 几乎在相同位置检测到手势
                    personpre['record'] = True

                    if f'record_hand_{pospre}' not in personpre:
                        # 当前手势不在前一帧的待处理列表中
                        personpre[f'record_hand_{pospre}'] = [personpre[f'hand_class_{pospre}']]
                    else:
                        # 当前手势在前一帧的待处理列表中
                        personpre[f'record_hand_{pospre}'].append(personpre[f'hand_class_{pospre}'])

                else:
                    # 发生漏检 误检导致手势位置偏移
                    if f'record_hand_{pospre}' not in personpre:
                        # 当前手势不在前一帧的待处理列表中
                        personpre[f'record_hand_{pospre}'] = ['N']
                    else:
                        # 当前手势在前一帧的待处理列表中
                        personpre[f'record_hand_{pospre}'].append('N')
            else:
                # 发生漏检
                pass
    return personpre


def add_trigger1(person_list, nc, frame_id, ges_len=5):
    # 首先对person_list 进行预处理，清除list中没有手框的
    tmp_person = []
    for index, person in enumerate(person_list):
        # 先判断手框是否在person中
        if 'hand' not in person:
            continue
        tmp_person.append(person)
    new_person_list = tmp_person

    # nc是前一帧的检测结果
    # 如果nc为空，用当前帧的结果进行赋值
    if not nc:
        nc = new_person_list

    # 用于确定nc中的元素都包含正类别
    positive_nc = list()
    for itm in nc:
        if 'relpos' in itm:
            pos_bool = list()
            for pos in itm['relpos']:
                pos_bool.append(check_label(itm[f'hand_class_{pos}']))
                itm = add_frame_id(itm, frame_id)
            if True in pos_bool:
                positive_nc.append(itm)
    nc = positive_nc
    if not nc:
        return nc
    # 增加自动补齐功能，左右手没检测到的时候pad[0 for _ in range(5)]
    nc = padding_zero(nc)

    # 对前后两帧的检测结果进行处理
    # 首先判断前一帧和当前帧有效手势的识别区域是否重叠度大于0.95
    # 如果重叠度大于0.95则进一步判断手框的iou
    # 如果手框的iou大于0.95则认为当前帧和前一帧手框未移动
    tmp_record_nc = list()
    for personnow in new_person_list:
        if 'effect_hand_area' in personnow:
            now_effect_hand = personnow['effect_hand_area']
            for personpre in nc:
                pre_effect_hand = personpre['effect_hand_area']

                effect_iou = box_iou(torch.tensor([now_effect_hand]), torch.tensor([pre_effect_hand]))
                if effect_iou > 0.95:
                    # 判断手框的iou
                    tmp_nc = limit_hand_area(personpre, personnow)
                else:
                    pass

    # 记录不满五帧的，使用’N‘强制补齐

    # 判断手势结果
    for index, hand in enumerate(nc):
        # if 'act' in hand:
        hand['act'] = False
        if 'hand_gesture' in hand:
            gesture = hand['hand_gesture']
            if len(gesture) >= ges_len:
                left, right, left_color, right_color = get_hand_gesture_array(gesture)
                hand['left'] = left
                hand['right'] = right
                hand['left_color'] = left_color
                hand['right_color'] = right_color
    return nc


# {'head_bbox': [477, 289, 756, 695, 0.85139], 'effect_hand_area': [0, 86, 1280, 720],
# 'hand': [[73, 368, 368, 680, 0.951]], 'odis': [397.290825466685], 'relpos': ['right'],
# 'hand_bbox_right': [73, 368, 368, 680, 0.951], 'color_type': 0, 'hand_class_right': 'P',
# 'act': False, 'hand_gesture': [['', 'P'], ['', 'P'], ['', 'P'], ['', 'P'], ['', 'P']], 'left': 'N', 'right': 'P'}

def get_hand_gesture_array(gesture, det_thre=0.6):
    label_list = ['N', 'F', 'P', 'L']

    det_len = len(gesture)
    left_n = 0
    left_f = 0
    left_p = 0
    left_l = 0

    right_n = 0
    right_f = 0
    right_p = 0
    right_l = 0

    for item in gesture:
        if item[0] == 'left' or item[0] == 'N':
            left_n += 1
        elif item[0] == 'F':
            left_f += 1
        elif item[0] == 'P':
            left_p += 1
        elif item[0] == 'L':
            left_l += 1

        if item[1] == 'right' or item[1] == 'N':
            right_n += 1
        elif item[1] == 'F':
            right_f += 1
        elif item[1] == 'P':
            right_p += 1
        elif item[1] == 'L':
            right_l += 1
    left = [left_n, left_f, left_p, left_l]
    right = [right_n, right_f, right_p, right_l]

    res_left = 'N'
    res_right = 'N'

    for index, (l, r) in enumerate(zip(left, right)):
        if l / det_len >= det_thre:
            res_left = label_list[index]

        if r / det_len >= det_thre:
            res_right = label_list[index]

    left_color = label_list.index(res_left) * 2
    right_color = label_list.index(res_right) * 2
    return res_left, res_right, left_color, right_color


def ges_label(nc_ch):
    label_list = ['F', 'P', 'L']
    res = False
    if 'relpos' in nc_ch:
        for item in nc_ch['relpos']:
            if f'hand_class_{item}' in nc_ch:
                if nc_ch[f'hand_class_{item}'] in label_list:
                    res = res | True
    return res


def cal_res_hand(person_list, nc):
    for index, person in enumerate(person_list):
        if 'effect_hand_area' in person:
            effect_hand = person['effect_hand_area']
            for item in nc:
                record_hand = list()
                nc_area = item['effect_hand_area']
                iou = box_iou(torch.tensor([effect_hand]), torch.tensor([nc_area]))
                if iou > 0.95:
                    # 找到了连续帧
                    if 'relpos' in person:
                        for posp in person['relpos']:
                            for posn in nc['relpos']:
                                iou_hand = box_iou(torch.tensor(person[f'hand_class_{posn}']),
                                                   torch.tensor(nc[f'hand_class_{posp}']))
                                if iou_hand > 0.95:
                                    # 手没有移动
                                    item['act'] = True
                                    if 'hand_class_left' in person:
                                        record_hand.append(person['hand_class_left'])
                                    else:
                                        record_hand.append('')

                                    if 'hand_class_right' in person:
                                        record_hand.append(person['hand_class_right'])
                                    else:
                                        record_hand.append('')

                                    if 'hand_gesture' in item:
                                        item['hand_gesture'].append(record_hand)
                                    else:
                                        item['hand_gesture'] = [record_hand]

                                    person['act'] = True

    pass


def parase_now_person(person_list, nc, frame_id):
    need_init = False
    # 首先先排除没有检测到手的person
    tmp_person = []
    for index, person in enumerate(person_list):
        # 先判断手框是否在person中
        if 'hand' not in person:
            continue
        if 'frame_count' not in person:
            person['frame_count'] = [frame_id]
        else:
            person['frame_count'].append(frame_id)
        tmp_person.append(person)
    person_list = tmp_person

    if not nc:
        # 初始化nc
        nc = person_list
        need_init = True

    if need_init:
        # 需要初始化，此时nc==person_list
        tmp_plist = list()
        record_hand = list()
        for person in nc:
            if 'hand_class_left' in person:
                record_hand.append(person['hand_class_left'])
            else:
                record_hand.append('')

            if 'hand_class_right' in person:
                record_hand.append(person['hand_class_right'])
            else:
                record_hand.append('')

            if record_hand[0] == record_hand[1] == '':
                return []
            if 'hand_gesture' in nc:
                person['hand_gesture'].append(record_hand)
            else:
                person['hand_gesture'] = [record_hand]
            person['act'] = True
            tmp_plist.append(person)
        nc = tmp_person
    else:
        # 此时nc已经记录上一帧的检测结果
        tmp_plist = list()
        for index, person in enumerate(person_list):
            # 先判断手框是否在person中
            if 'hand' not in person:
                continue
            effect_hand = person['effect_hand_area']
            for item in nc:
                nc_area = item['effect_hand_area']

                effect_iou = box_iou(torch.tensor([effect_hand]), torch.tensor([nc_area]))
                if effect_iou > 0.95:
                    # 前一帧和当前帧的位移较小
                    def limit_hand_area(personpre, personaft):
                        for pospre in personpre['relpos']:
                            for posaft in personaft['relpos']:
                                if posaft == pospre:
                                    hand_bbox_pre = personpre[f'hand_bbox_{pospre}']
                                    hand_bbox_aft = personaft[f'hand_bbox_{posaft}']
                                    hand_iou = box_iou(torch.tensor([hand_bbox_pre][:4]),
                                                       torch.tensor([hand_bbox_aft][:4]))
                                    if hand_iou > 0.95:
                                        pass
                                    elif 0.9 < hand_iou <= 0.95:
                                        pass
                                    else:
                                        pass

                        pass

                    pass

                elif 0.9 < effect_iou <= 0.95:
                    # 前一帧和当前帧的位移稍小
                    pass


gesture_bucket = {}
for i in range(50):
    gesture_bucket[i] = []

GESTURE_BUCKET = gesture_bucket


def incream_bucket(bucket):
    for k, v in bucket.items():
        if not v:
            return k


def parase_now_person1(person_list, nc, frame_id, hand_ges_bucket=None):
    global GESTURE_BUCKET

    need_init = False
    # 首先先排除没有检测到手的person
    tmp_person = []
    for index, person in enumerate(person_list):
        # 先判断手框是否在person中
        if 'hand' not in person:
            continue
        if 'frame_count' not in person:
            person['frame_count'] = [frame_id]
        else:
            person['frame_count'].append(frame_id)
        tmp_person.append(person)
    person_list = tmp_person

    if not nc:
        # 初始化nc
        nc = person_list
        need_init = True

    tmp_plist = list()
    for index, person in enumerate(person_list):
        # 先判断手框是否在person中
        if 'hand' not in person:
            continue
        effect_hand = person['effect_hand_area']
        for item in nc:
            nc_area = item['effect_hand_area']

            effect_iou = box_iou(torch.tensor([effect_hand]), torch.tensor([nc_area]))
            if effect_iou > 0.95:
                if 'num' not in person:
                    num = incream_bucket(GESTURE_BUCKET)
                    person['num'] = num

            else:
                pass


def main(args):
    """Process the input with all modules frame by frame.
    Args:
        args (dict): The parsed command line arguments.
    """
    cfg = Config.fromfile(args.config)
    modules = build_modules(cfg.modules)
    visualizers = build_visualizers(cfg.visualizers)

    # init file reader and video cap
    f_inputs = open(args.input, 'r') if args.input else None

    # prepare attribute info for output file
    f_outputs = None
    if args.output:
        attributes = gather_attributes(args.input, cfg.modules)
        f_outputs = open(args.output, 'w')
        f_outputs.write('{}\n'.format(json.dumps({"attributes": attributes})))

    # for video or camera
    dataset = LoadUSBcam(pipe=args.video, flip=args.flip, frameWidth=args.w, frameHeight=args.h)

    size = dataset.getFrameSize()
    fps = dataset.getFps()

    ori_video_output_path = os.path.join(args.video_data, 'data', 'origin')
    if not os.path.exists(ori_video_output_path):
        os.makedirs(ori_video_output_path)
    proc_video_output_path = os.path.join(args.video_data, 'data', 'process')
    if not os.path.exists(proc_video_output_path):
        os.makedirs(proc_video_output_path)

    file_name = "_".join(''.join(random.choice(string.ascii_lowercase) for i in range(8)) for _ in range(4))
    ori_video = os.path.join(ori_video_output_path, file_name + '_ori.mp4')
    proc_video = os.path.join(proc_video_output_path, file_name + '_proc.mp4')
    ori_vw = cv2.VideoWriter(ori_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    proc_vw = cv2.VideoWriter(proc_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    nc = []

    for idx, img in enumerate(dataset):
        print("frame_id: ", idx)
        ori_img = img.copy()
        # process frame module by model
        persons = []
        for module_idx, module in enumerate(modules):
            persons = module.process(0, img, persons)
        persons = persons
        nc = add_trigger1(persons, nc, idx)

        # nc = parase_now_person1(persons, nc, idx)
        if args.show:
            for visualizer_idx, visualizer in enumerate(visualizers):
                if visualizer_idx >= len(visualizers) - 2:
                    img = visualizer.draw(idx, img, nc)
                elif visualizer_idx == 1:
                    img = visualizer.draw(idx, img, persons)
                else:
                    img = visualizer.draw(idx, img, persons)

            if args.show:
                cv2.imshow("result", img)
                cv2.waitKey(30)
        if ori_vw:
            ori_vw.write(ori_img)

        if proc_vw:
            proc_vw.write(img)

    if ori_vw:
        ori_vw.release()

    if proc_vw:
        proc_vw.release()


if __name__ == "__main__":
    args = parse_argument()

    modify_opt = {
        'config': 'configs/hand_gesture.py',
        'video': r'F:\new_project1\test\man.mp4',
        # 'video': r'0',
        'show': True,

    }

    for key, value in modify_opt.items():
        setattr(args, key, value)
    logger = get_logger(name='base', log_level="info")
    # cuda env
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # check flow inputs/outputs
    assert check_config_file(args.config, args.video,
                             args.input), "check config file fail."
    main(args)
# --hidden-import yaml --hidden-import thop --hidden-import seaborn --hidden-import setuptools_scm --hidden-import PIL
