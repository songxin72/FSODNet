#coding=utf-8

import os
import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset
import imageio
import torch
import torchvision.transforms.functional as TF
from models.common import RGB2YCrCb, clamp
import h5py
from glob import glob
to_tensor = transforms.Compose([transforms.ToTensor()])
from imutils import paths
from PIL import Image
import customTransform


def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value  is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        # flatten=True Read the image as a grayscale map.
        return imageio.imread(path, as_gray=True, pilmode='YCbCr').astype(np.float)
    else:
        return imageio.imread(path, pilmode='YCbCr').astype(np.float)


# BGR
mean_rgb = np.array([[[0.551 * 255, 0.619 * 255, 0.532 * 255]]])
mean_inf = np.array([[[0.341 * 255,  0.360 * 255, 0.753 * 255]]])
std_rgb = np.array([[[0.241 * 255, 0.236 * 255, 0.244 * 255]]])
std_inf = np.array([[[0.208 * 255, 0.269 * 255, 0.241 * 255]]])


class Data(Dataset):
    def __init__(self, root, mode='train'):
        if mode == 'train':
            self.samples = []
            lines = os.listdir(os.path.join(root, 'Inf'))
            for line in lines:
                inf_path = root + '/' + 'Inf' + '/' + line       # os.path.join(root, 'Inf', line)
                maskpath = root + '/' + 'GT' + '/' + line        # os.path.join(root, 'GT', line)
                rgb_path = root + '/' + 'RGB' + '/' + line
                self.samples.append([rgb_path, inf_path, maskpath])

            self.transform = transform.Compose(transform.Normalize(mean=mean_inf, std=std_inf),
                                               transform.Resize(480, 640),
                                               # transform.RandomHorizontalFlip(),# if need
                                               transform.ToTensor())
            self.transform2 = transform.Composes(transform.Normalizes(mean=mean_rgb, std=std_rgb),
                                               transform.Resizes(480, 640),
                                               transform.ToTensors())

    def __getitem__(self, idx):
        rgb_path, inf_path, maskpath = self.samples[idx]
        rgb = cv2.imread(rgb_path).astype(np.float32)
        inf = cv2.imread(inf_path).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        inf, mask = self.transform(inf, mask)
        rgb = self.transform2(rgb)
        return maskpath.split('/')[-1], rgb, inf, mask

    def __len__(self):
        return len(self.samples)


# 集成Dataset类
class OurDataset(Dataset):
    def __init__(self):
        traindir = os.path.join(r'datasets/vaisTrain')
        secdir = os.path.join(r'datasets/ifTrain')
        vi_data, if_data = get_semi_data(traindir, secdir)
        self.data_info = vi_data
        self.sec_info = if_data
        valid_augmentation = transform.Composes(transform.Resizes(224, 224),
                                           transform.ToTensors())
        self.transform = valid_augmentation
        self.target_transform = valid_augmentation

    def __getitem__(self, index):
        path_img, label = self.data_info[index]  # 通过index索引返回一个图像路径fn 与 标签label
        path_sec, label_sec = self.sec_info[index]
        img = cv2.imread(path_img).astype(np.float32)
        sec = cv2.imread(path_sec).astype(np.float32)
        name = path_img.split('/')[-1]
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.target_transform(sec)
        return name, img1, img2, label  # 这就返回一个样本

    def __len__(self):
        return len(self.data_info)  # 返回长度，index就会自动的指导读取多少


def get_loader(config):
    # dataset = Train_roadDataset()
    # dataset = Data('./road')
    # dataset = OurDataset()
    dataset = Train_roadsceneDataset()
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader


class Train_roadDataset(data.Dataset):
    def __init__(
            self,
    ):
        """At this moment, only test set is considered."""
        super(Train_roadDataset, self).__init__()

        self.test_path = "./Test_images/InfVis/Inf"
        self.visible_path = "./Test_images/InfVis/Vis"
        self.ir_mask_path = "./Test_images/InfVis/SOSmask"
        self.p_train_imgs = os.listdir(self.visible_path)
        self.p_train_inf_imgs = os.listdir(self.test_path)
        self.p_train_gts = os.listdir(self.ir_mask_path)
        self.p_train_masks = None
        self.p_train_vars = None
        self.p_crf_masks = None
        self.name = "tno"
        self.size = len(self.p_train_imgs)

    def __len__(self):
        return len(self.p_train_imgs)

    def __getitem__(self, index: int) -> dict:


        inf_path = os.path.join(self.test_path, self.p_train_inf_imgs[index])
        input_ir = imread(inf_path)
        input_ir = np.reshape(input_ir, [1, input_ir.shape[0], input_ir.shape[1]])

        vis_path = os.path.join(self.visible_path, self.p_train_imgs[index])
        input_vi = imread(vis_path)
        input_vi = np.reshape(input_vi, [1, input_vi.shape[0], input_vi.shape[1]])

        image = input_vi
        inf_image = input_ir

        filename = self.p_train_imgs[index].split('/')[0]

        gt_path = os.path.join(self.ir_mask_path, self.p_train_gts[index])
        input_mk = imread(gt_path)
        input_mk = np.reshape(input_mk, [1, input_mk.shape[0], input_mk.shape[1]])
        input_mk = input_mk / 255

        masks = torch.from_numpy(input_mk)

        image = TF.to_tensor(image).float()  # TF.normalize(TF.to_tensor(image), self.mean, self.std)  # 源代码
        # image = clamp(image)
        image = image.permute(1, 2, 0)
        # image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = TF.to_tensor(inf_image).float()
        # inf_image = clamp(inf_image)
        inf_image = inf_image.permute(1, 2, 0)
        # inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        # masks = torch.tensor(np.expand_dims(masks, 0)).float()
        masks = torch.tensor(masks).float()

        # # masks = np.asarray(masks, np.int64)
        # if masks.max() > 1.0:
        #     masks = masks > 0
        # masks = masks.permute(2, 1, 0)

        return filename, image, inf_image, masks


class Train_roadsceneDataset(data.Dataset):

    def __init__(
            self,
    ):
        """At this moment, only test set is considered."""
        super(Train_roadsceneDataset, self).__init__()

        self.test_path = "./IF"
        self.visible_path = "./VIS"
        self.ir_mask_path = "./road_masks"
        self.p_train_imgs = os.listdir(self.visible_path)
        self.p_train_inf_imgs = os.listdir(self.test_path)
        self.p_train_gts = os.listdir(self.ir_mask_path)
        self.p_train_masks = None
        self.p_train_vars = None
        self.p_crf_masks = None
        self.name = "tno"
        self.size = len(self.p_train_imgs)

    def __len__(self):
        return len(self.p_train_imgs)

    def __getitem__(self, index: int) -> dict:

        padding = 0
        inf_path = os.path.join(self.test_path, self.p_train_inf_imgs[index])
        input_ir = imread(inf_path)
        # input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        input_ir = np.reshape(input_ir, [1, input_ir.shape[0], input_ir.shape[1]])

        vis_path = os.path.join(self.visible_path, self.p_train_imgs[index])
        input_vi = imread(vis_path)
        # input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        input_vi = np.reshape(input_vi, [1, input_vi.shape[0], input_vi.shape[1]])

        image = input_vi
        inf_image = input_ir

        filename = self.p_train_imgs[index].split('/')[0]

        gt_path = os.path.join(self.ir_mask_path, self.p_train_gts[index])
        input_mk = imread(gt_path)
        # input_mk = np.lib.pad(input_mk, ((padding, padding), (padding, padding)), 'edge')
        input_mk = np.reshape(input_mk, [1, input_mk.shape[0], input_mk.shape[1]])

        masks = torch.from_numpy(input_mk)

        image = TF.to_tensor(image).float()  # TF.normalize(TF.to_tensor(image), self.mean, self.std)  # 源代码
        image = image.permute(1, 2, 0)
        image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = TF.to_tensor(inf_image).float()
        inf_image = inf_image.permute(1, 2, 0)
        inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        masks = torch.tensor(masks).float()

        return filename, image, inf_image, masks


class Test_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        # self.images = get_image_list(name, config, 'test')  # self.gts
        self.test_path = "./Test_images/InfVis/Inf"
        self.visible_path = "./Test_images/InfVis/Vis"
        self.ir_mask_path = "./Test_images/InfVis/SOSmask"
        self.p_train_imgs = os.listdir(self.visible_path)
        self.p_train_inf_imgs = os.listdir(self.test_path)
        self.p_train_gts = os.listdir(self.ir_mask_path)
        self.size = len(self.p_train_imgs)
        self.dataset_name = name

    def load_data(self, index):
        inf_path = os.path.join(self.test_path, self.p_train_inf_imgs[index])
        input_ir = imread(inf_path)
        input_ir = np.reshape(input_ir, [1, input_ir.shape[0], input_ir.shape[1]])

        vis_path = os.path.join(self.visible_path, self.p_train_imgs[index])
        input_vi = imread(vis_path)
        input_vi = np.reshape(input_vi, [1, input_vi.shape[0], input_vi.shape[1]])

        image = input_vi
        inf_image = input_ir

        name = self.p_train_imgs[index].split('/')[0]

        gt_path = os.path.join(self.ir_mask_path, self.p_train_gts[index])
        input_mk = imread(gt_path)
        input_mk = np.reshape(input_mk, [1, input_mk.shape[0], input_mk.shape[1]])
        input_mk = input_mk / 255
        masks = torch.from_numpy(input_mk)

        image = TF.to_tensor(image).float()  # TF.normalize(TF.to_tensor(image), self.mean, self.std)  # 源代码
        # image = clamp(image)
        image = image.permute(1, 2, 0)
        image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = TF.to_tensor(inf_image).float()
        # inf_image = clamp(inf_image)
        inf_image = inf_image.permute(1, 2, 0)
        inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        return image, inf_image, masks, name  # gt, name


class Testroadsecne_Dataset:

    def __init__(self, name, config=None):
        self.config = config
        # self.images = get_image_list(name, config, 'test')  # self.gts
        self.test_path = "./IF"
        self.visible_path = "./VIS"
        self.ir_mask_path = "./road_masks"
        self.p_train_imgs = os.listdir(self.visible_path)
        self.p_train_inf_imgs = os.listdir(self.test_path)
        self.p_train_gts = os.listdir(self.ir_mask_path)
        self.size = len(self.p_train_imgs)
        self.dataset_name = name

    def load_data(self, index):
        padding = 0
        inf_path = os.path.join(self.test_path, self.p_train_inf_imgs[index])
        input_ir = imread(inf_path) #/ 127.5 - 1.0
        #input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        input_ir = np.reshape(input_ir, [1, input_ir.shape[0], input_ir.shape[1]])

        vis_path = os.path.join(self.visible_path, self.p_train_imgs[index])
        input_vi = imread(vis_path) #/ 127.5 - 1.0
        #input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        input_vi = np.reshape(input_vi, [1, input_vi.shape[0], input_vi.shape[1]])

        image = input_vi
        inf_image = input_ir

        name = self.p_train_imgs[index].split('/')[0]

        gt_path = os.path.join(self.ir_mask_path, self.p_train_gts[index])
        input_mk = imread(gt_path) #/ 127.5 - 1.0
        #input_mk = np.lib.pad(input_mk, ((padding, padding), (padding, padding)), 'edge')
        input_mk = np.reshape(input_mk, [1, input_mk.shape[0], input_mk.shape[1]])
        masks = torch.from_numpy(input_mk)

        image = TF.to_tensor(image).float()  # TF.normalize(TF.to_tensor(image), self.mean, self.std)  # 源代码
        image = image.permute(1, 2, 0)
        image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = TF.to_tensor(inf_image).float()
        inf_image = inf_image.permute(1, 2, 0)
        inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        return image, inf_image, masks, name


class TestData:
    def __init__(self, name, config=None):
        root = './road'
        self.config = config
        self.samples = []
        lines = os.listdir(os.path.join(root, 'Inf'))
        for line in lines:
            rgb_path = root + '/' + 'RGB' + '/' + line
            inf_path = root + '/' + 'Inf' + '/' + line
            maskpath = root + '/' + 'GT' + '/' + line
            self.samples.append([rgb_path, inf_path, maskpath])

        self.transform = transform.Compose(transform.Normalize(mean=mean_inf, std=std_inf),
                                           transform.Resize(480, 640),
                                           transform.ToTensor())
        self.transform2 = transform.Composes(transform.Normalizes(mean=mean_rgb, std=std_rgb),
                                            transform.Resizes(480, 640),
                                            transform.ToTensors())
        self.size = len(self.samples)
        self.dataset_name = name

    def load_data(self, idx):
        rgb_path, inf_path, maskpath = self.samples[idx]
        rgb = cv2.imread(rgb_path).astype(np.float32)
        inf = cv2.imread(inf_path).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        inf, mask = self.transform(inf, mask)
        rgb = self.transform2(rgb)
        rgb = torch.unsqueeze(rgb, 0)
        inf = torch.unsqueeze(inf, 0)
        return rgb, inf, mask, maskpath.split('/')[-1]


class TestDataset:
    def __init__(self, name, config=None):
        self.config = config
        traindir = os.path.join(r'datasets/vaisTrain')
        secdir = os.path.join(r'datasets/ifTrain')
        vi_data, if_data = get_semi_data(traindir, secdir)
        self.data_info = vi_data
        self.sec_info = if_data
        valid_augmentation = transform.Composes(transform.Resizes(224, 224),
                                           transform.ToTensors())
        self.transform = valid_augmentation
        self.target_transform = valid_augmentation
        self.size = len(self.data_info)
        self.dataset_name = name

    def load_data(self, index):
        path_img, label = self.data_info[index]  # 通过index索引返回一个图像路径fn 与 标签label
        path_sec, label_sec = self.sec_info[index]
        img = cv2.imread(path_img).astype(np.float32)  # 把图像转成RGB
        sec = cv2.imread(path_sec).astype(np.float32)
        name = path_img.split('/')[-1]
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.target_transform(sec)
        img1 = torch.unsqueeze(img1, 0)
        img2 = torch.unsqueeze(img2, 0)
        return img1, img2, label, name  # 这就返回一个样本


def get_tnoloader(config):
    # dataset = TNO_data('data')  #
    dataset = Train_tnoDataset()
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader


class TNO_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        data_dir_ir = os.path.join('./{}'.format(data_dir), "Train_ir", "train.h5")
        data_dir_vi = os.path.join('./{}'.format(data_dir), "Train_vi", "train.h5")
        train_data_ir_mask = os.path.join('./{}'.format(data_dir), "tno_mask", "train.h5")
        with h5py.File(data_dir_ir, 'r') as hf:
            self.data_ir = np.array(hf.get('data'))
        with h5py.File(data_dir_vi, 'r') as hf:
            self.data_vi = np.array(hf.get('data'))
        with h5py.File(train_data_ir_mask, 'r') as hf:
            self.data_mask = np.array(hf.get('data'))
        # self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        print(self.data_ir.shape, self.data_vi.shape)
        self.transform = transform

    def __getitem__(self, index):
        # name = self.name_list[index]  # 获得当前图片的名称

        # inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  # 获取红外图像
        inf_image = self.data_ir[index]
        vis_image = self.data_vi[index]
        mask_image = self.data_mask[index]
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        # vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        vis_y_image = clamp(vis_image)
        vis_cb_image = clamp(vis_image)
        vis_cr_image = clamp(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, mask_image

    def __len__(self):
        return self.data_ir.shape[0]


class Train_tnoDataset(data.Dataset):
    def __init__(
            self,
    ):
        """At this moment, only test set is considered."""
        super(Train_tnoDataset, self).__init__()

        self.test_path = "./Test_irs"
        self.visible_path = "./Test_vis"
        self.ir_mask_path = "./Test_mask"
        self.p_train_imgs = os.listdir(self.visible_path)
        self.p_train_inf_imgs = os.listdir(self.test_path)
        self.p_train_gts = os.listdir(self.ir_mask_path)
        self.p_train_masks = None
        self.p_train_vars = None
        self.p_crf_masks = None
        self.name = "tno"
        self.size = len(self.p_train_imgs)

    def __len__(self):
        return len(self.p_train_imgs)

    def __getitem__(self, index: int) -> dict:
        padding = 0
        inf_path = os.path.join(self.test_path, self.p_train_inf_imgs[index])
        input_ir = imread(inf_path) / 127.5 - 1.0
        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([1, w, h])

        vis_path = os.path.join(self.visible_path, self.p_train_imgs[index])
        input_vi = imread(vis_path) / 127.5 - 1.0
        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([1, w, h])

        image = input_vi
        inf_image = input_ir

        filename = self.p_train_imgs[index].split('/')[0]

        gt_path = os.path.join(self.ir_mask_path, self.p_train_gts[index])
        input_mk = imread(gt_path) / 127.5 - 1.0
        input_mk = np.lib.pad(input_mk, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_mk.shape
        input_mk = input_mk.reshape([1, w, h])

        masks = torch.from_numpy(input_mk)

        image = TF.to_tensor(image).float()  # TF.normalize(TF.to_tensor(image), self.mean, self.std)  # 源代码
        image = clamp(image)
        image = image.permute(1, 2, 0)
        # image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = TF.to_tensor(inf_image).float()
        inf_image = clamp(inf_image)
        inf_image = inf_image.permute(1, 2, 0)
        # inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        # masks = torch.tensor(np.expand_dims(masks, 0)).float()
        masks = torch.tensor(masks).float()

        # # masks = np.asarray(masks, np.int64)
        # if masks.max() > 1.0:
        #     masks = masks > 0
        # masks = masks.permute(2, 1, 0)

        return filename, image, inf_image, masks


class TNO_test:
    def __init__(self, transform=to_tensor):
        super().__init__()
        dataset_ir = r'./Test_irs'
        dataset_vi = r'./Test_vis'
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset_ir)))
        self.data_ir = glob(os.path.join(data_dir, "*.png"))
        self.data_ir.extend(glob(os.path.join(data_dir, "*.bmp")))
        self.data_ir.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))

        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset_vi)))
        self.data_vi = glob(os.path.join(data_dir, "*.png"))
        self.data_vi.extend(glob(os.path.join(data_dir, "*.bmp")))
        self.data_vi.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))

        self.p_train_gts = sorted(list(paths.list_images("./Test_mask")))
        self.p_train_gts.sort(key=lambda x: int((x.split('/')[-1]).split('.')[0]))

        self.size = len(self.data_ir)
        self.transform = transform

    def load_data(self, index):

        padding = 0
        input_ir = imread(self.data_ir[index]) / 127.5 - 1.0  # self.imread(self.data_ir[index]) / 255 #
        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([w, h, 1])

        input_vi = imread(self.data_vi[index]) / 127.5 - 1.0  # (self.imread(self.data_vi[index]) - 127.5) / 127.5#
        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([w, h, 1])

        inf_image = input_ir
        vis_image = input_vi
        inf_image = self.transform(inf_image).float()
        vis_image = self.transform(vis_image).float()
        vis_image = clamp(vis_image)
        # vis_image = vis_image.permute(1, 2, 0)
        vis_image = torch.tensor(np.expand_dims(vis_image, 0)).float()
        inf_image = clamp(inf_image)
        # inf_image = inf_image.permute(1, 2, 0)
        inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        name = self.data_ir[index].split('/')[-1].split('.')[0]

        # vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        vis_y_image = clamp(vis_image)
        vis_cb_image = clamp(vis_image)
        vis_cr_image = clamp(vis_image)

        input_mk = imread(self.p_train_gts[index]) / 127.5 - 1.0
        input_mk = np.lib.pad(input_mk, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_mk.shape
        input_mk = input_mk.reshape([w, h])
        masks = torch.from_numpy(input_mk)

        return vis_image, inf_image, masks, name

    def __len__(self):
        return len(self.data_ir)


def get_semi_data(dir, sec_dir):
    n_class = 6
    img_num_list = ['5', '2', '1', '0', '3', '4']
    data_info = list()
    sec_info = list()
    data_labels = list()
    sec_labels = list()
    for root, dirs, _ in os.walk(dir):
        # 遍历类别
        for key in range(0, 6):
            for sub_dir in dirs:
                if (sub_dir == img_num_list[key]):
                    img_names = os.listdir(os.path.join(root, sub_dir))
                    img_names = list(filter(lambda x: x.endswith('.png'), img_names))

                    # 遍历图片
                    for i in range(len(img_names)):
                        img_name = img_names[i]
                        path_img = os.path.join(root, sub_dir, img_name)
                        label = int(str(key))
                        data_labels.append(int(label))
                        data_info.append((path_img, int(label)))
                    # print('imbalanced', len(data_info))
                    break
        break
    for root, dirs, _ in os.walk(sec_dir):
        # 遍历类别
        for key in range(0, 6):
            for sub_dir in dirs:
                if (sub_dir == img_num_list[key]):
                    img_names = os.listdir(os.path.join(root, sub_dir))
                    img_names = list(filter(lambda x: x.endswith('.png'), img_names))

                    # 遍历图片
                    for i in range(len(img_names)):
                        img_name = img_names[i]
                        path_img = os.path.join(root, sub_dir, img_name)
                        label = int(str(key))
                        sec_labels.append(label)
                        sec_info.append((path_img, int(label)))
                    # print('imbalanced', len(data_info))
                    break
        break
    data_labels = np.array(data_labels)
    # n_labels = 40
    # data_x, data_u = list(), list()
    # sec_x, sec_u = list(), list()
    # for i in range(n_class):
    #     indices = np.where(data_labels == i)[0]
    #     np.random.shuffle(indices)
    #     inds_x, inds_u = indices[:n_labels], indices[n_labels:]
    #     for j in inds_x:
    #         data_x.append(data_info[j])
    #         sec_x.append(sec_info[j])
    #     for j in inds_u:
    #         data_u.append(data_info[j])
    #         sec_u.append(sec_info[j])
    return data_info, sec_info  ## 返回的也就是图像路径 和 标签
