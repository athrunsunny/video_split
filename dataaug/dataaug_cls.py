import os
import shutil

import cv2
import PIL.Image as Image
import torchvision.transforms
from torchvision import transforms
import torch
from glob import glob
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class DataAugmentation(object):
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    def __init__(self):
        super(DataAugmentation, self).__init__()
        self.transforms = transforms

    def add_gasuss_noise(self, img, mean=0, std=0.05):
        """
        随机高斯噪声
        :param img: Image
        :param boxes: bbox坐标
        :param mean:
        :param std:
        :return:
        """
        img = self.to_tensor(img)
        noise = torch.normal(mean, std, img.shape)
        img += noise
        img = img.clamp(min=0, max=1.0)
        return self.to_image(img)

    def add_salt_noise(self, img):
        """
        随机盐噪声
        :param img: Image
        :param boxes: bbox坐标
        :return:
        """
        img = self.to_tensor(img)
        noise = torch.rand(img.shape)
        alpha = np.random.random()
        img[noise[:, :, :] > alpha] = 1.0
        return self.to_image(img)

    def add_pepper_noise(self, img):
        """
        随机椒噪声
        :param img: Image
        :param boxes: bbox坐标
        :return:
        """
        img = self.to_tensor(img)
        noise = torch.rand(img.shape)
        alpha = np.random.random()
        img[noise[:, :, :] > alpha] = 0
        return self.to_image(img)

    def crop(self, img, size=(108, 108)):
        crop_obj = torchvision.transforms.CenterCrop(size)
        image = crop_obj(img)
        return image

    def gaussian_blur(self, img, kernel_size=5, sigma=(0.1, 2.0)):
        transform = self.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        img = transform(img)
        return img


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    try:
        # Resize and pad image while meeting stride-multiple constraints
        # im = im[:,240:1680,:]
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_AREA)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        # cv2.imshow('ddd', im)
        # cv2.imwrite(save_path, im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return im, ratio, (dw, dh)
    except:
        return None, None, None


def add_noise(file_path, noise_prob=0.3, crop_prob=1):
    aug = DataAugmentation()
    if isinstance(file_path, str):
        image = Image.open(file_path)
    else:
        if isinstance(file_path, np.ndarray):
            # image = Image.fromarray(cv2.cvtColor(file_path, cv2.COLOR_BGR2RGB))
            image = Image.fromarray(file_path)
        else:
            image = file_path
    # add noise and crop
    first_prob = np.random.random()
    if first_prob > noise_prob:
        image = aug.add_gasuss_noise(image)

    second_prob = np.random.random()
    if second_prob > crop_prob:
        image = aug.crop(image)
    return image


def add_blur(file_path, gass_prob=0.1):
    aug = DataAugmentation()
    if isinstance(file_path, str):
        image = Image.open(file_path)
    else:
        if isinstance(file_path, np.ndarray):
            # image = Image.fromarray(cv2.cvtColor(file_path, cv2.COLOR_BGR2RGB))
            image = Image.fromarray(file_path)
        else:
            image = file_path
    # add noise and crop
    prob = np.random.random()
    if prob > gass_prob:
        image = aug.gaussian_blur(image)
    return image


def add_padding(file_path, left_pad=15, top_pad=15):
    dim = (left_pad, left_pad, top_pad, top_pad)
    if isinstance(file_path, str):
        image = cv2.imread(file_path)
    else:
        #######
        if isinstance(file_path, np.ndarray):
            image = file_path
            # image = Image.fromarray(cv2.cvtColor(file_path, cv2.COLOR_BGR2RGB))
        else:
            # image = file_path
            # image = Image.fromarray(cv2.cvtColor(file_path, cv2.COLOR_BGR2RGB))
            image = cv2.cvtColor(np.array(file_path), cv2.COLOR_RGB2BGR)
            # image = np.array(file_path)

    X = torch.tensor(image).transpose(0, 2).transpose(2, 1)
    X = F.pad(X, dim, "constant", value=114).transpose(2, 1).transpose(0, 2)
    padX = X.data.numpy()
    return padX


def resize(file):
    scale = [0.1, 0.2, 0.3, 0.5, 0.66, 0.75]
    now_scale = np.random.choice(scale, p=[0.4, 0.25, 0.1, 0.1, 0.1, 0.05])
    if isinstance(file, str):
        image = cv2.imread(file)
    else:
        if isinstance(file, np.ndarray):
            image = file
        else:
            # image = Image.fromarray(cv2.cvtColor(file_path, cv2.COLOR_BGR2RGB))
            image = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)
            # image = np.array(file)
    img = cv2.resize(image, None, fx=now_scale, fy=now_scale)
    return img


def create_file(file_path,name):
    processed_path = os.path.join(file_path, name)
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    return processed_path


def save_image(image, image_name, des_path):
    filename = des_path.replace("\\", "/").split("/")[-1]
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = letterbox(image, 128)[0]
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = np.array(image)

    img_name = image_name.split('.')[0] + "_" + filename + '.jpg'
    des_path = os.path.join(des_path, img_name)
    cv2.imwrite(des_path, image)


def create_sample(file_path):  # oringin method
    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(file_path + "\\*." + extern))

    # seed = np.random.randint(1,100)
    np.random.seed(35)
    func_name = ['add_padding', 'resize', 'add_blur', 'add_noise']
    funct = [eval(func) for func in func_name]

    path_list = list()
    for image_file in tqdm(files):
        image_name = image_file.replace("\\", "/").split("/")[-1]
        image = cv2.imread(image_file)
        # image = letterbox(image, 224)[0]
        for index in range(len(funct)):
            processed_path = create_file(file_path,func_name[index])
            image = funct[index](image)

            if processed_path not in path_list:
                path_list.append(processed_path)

            save_image(image.copy(), image_name, processed_path)

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = letterbox(image, 128)[0]
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = np.array(image)
        img_name = image_name.split('.')[0] + "_last" + '.jpg'
        plast = create_file(file_path,'last')
        des_path = os.path.join(plast, img_name)
        cv2.imwrite(des_path, image)

        opath = create_file(file_path,'ori')
        shutil.move(image_file, opath)

    cpath = create_file(file_path,'choice')
    imfiles = list()
    for file in os.listdir(file_path):
        imfiles.extend(glob(file_path + f"\\{file}\\*.jpg"))
    print(len(imfiles))
    np.random.seed(31)
    for img in tqdm(imfiles):
        choice_prob = np.random.random()
        if choice_prob > 0.2:
            shutil.copy(img, cpath)


def create_single_func(file_path, func_name):
    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(file_path + "\\*." + extern))

    # seed = np.random.randint(1,100)
    np.random.seed(35)
    funct = eval(func_name)

    for image_file in tqdm(files):
        image_name = image_file.replace("\\", "/").split("/")[-1]
        image = cv2.imread(image_file)
        image = letterbox(image, 128)[0]

        processed_path = create_file(file_path,func_name)
        image = funct(image)
        save_image(image.copy(), image_name, processed_path)

        opath = create_file(file_path,'ori')
        shutil.move(image_file, opath)


def create_sample_all(file_path):
    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = list()
    for extern in externs:
        files.extend(glob(file_path + "\\*." + extern))

    # seed = np.random.randint(1,100)
    np.random.seed(35)
    #func_name = ['add_padding', 'add_noise', 'add_blur', 'add_blur',]# 'add_blur','add_noise']
    func_name = ['add_padding', 'add_noise', 'add_blur' ]

    funct = [eval(func) for func in func_name]

    path_list = list()
    for image_file in tqdm(files):
        image_name = image_file.replace("\\", "/").split("/")[-1]
        image = cv2.imread(image_file)
        image = letterbox(image, 64)[0]
        for index in range(len(funct)):
            processed_path = create_file(file_path,func_name[index])
            image = funct[index](image)

            if processed_path not in path_list:
                path_list.append(processed_path)

            save_image(image.copy(), image_name, processed_path)

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = letterbox(image, 128)[0]
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = np.array(image)
        img_name = image_name.split('.')[0] + "_last" + '.jpg'
        plast = create_file(file_path,'last')
        des_path = os.path.join(plast, img_name)
        cv2.imwrite(des_path, image)

        opath = create_file(file_path,'ori')
        shutil.move(image_file, opath)

    cpath = create_file(file_path,'choice')
    apath = create_file(file_path,'annother')
    imfiles = list()
    for file in os.listdir(file_path):
        imfiles.extend(glob(file_path + f"\\{file}\\*.jpg"))
    print(len(imfiles))
    np.random.seed(31)
    for img in tqdm(imfiles):
        choice_prob = np.random.random()
        if choice_prob > 0.2:
            shutil.move(img, cpath)
        else:
            shutil.move(img, apath)

def create_dataset(file_path):
    file_name = os.listdir(file_path)
    for file in file_name:
        dir = os.path.join(file_path,file)
        create_sample_all(dir)

if __name__ == "__main__":
    # 单个类的文件夹做处理 ['add_padding', 'resize', 'add_blur', 'add_noise']
    # D:\Users\yl3146\Desktop\videoprocess1\fist\ori\add_padding\ori\ori\add_noise\add_blur\add_blur\add_noise
    path = r'D:\Users\yl3146\Desktop\1208caiji'
    # create_sample(file_path)
    # create_single_func(file_path,'add_noise')
    # create_sample_all(file_path)
    create_dataset(path)
