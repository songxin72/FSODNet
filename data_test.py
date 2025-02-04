import os, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import imageio
from os.path import join
from glob import glob
from imageio import imsave
import cv2
from util import *
from metric import *
import pydensecrf.densecrf as dcrf

to_tensor = transforms.Compose([transforms.ToTensor()])
from imutils import paths

try:
    from . import transform
except:
    import transform

# mean = np.array((104.00699, 116.66877, 122.67892)).reshape((1, 1, 3))
mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def get_image_list(name, config, phase):
    images = []
    gts = []

    image_root = os.path.join(config['data_path'], name, 'images')
    if phase == 'train' and name == 'MSB-TR':
        tag = 'moco'
    else:
        tag = 'segmentations'

    print(tag)
    gt_root = os.path.join(config['data_path'], name, tag)

    images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.png')])  # 原来是jpg
    # gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])

    return images  # , gts


def crf_refine(img, annos):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    # print(img.shape[:2],annos.shape)
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')  # 创建和输入图片同样大小的U
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])  # 和输入图片同样大小
    return res.astype('uint8')


def get_loader(config):
    dataset = Train_Dataset(config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader


def random_light(x):
    contrast = np.random.rand(1) + 0.5
    light = np.random.randint(-20, 20)
    x = contrast * x + light
    return np.clip(x, 0, 255)


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


class Train_TNODataset(data.Dataset):
    def __init__(
            self,
    ):
        """At this moment, only test set is considered."""
        super(Train_TNODataset, self).__init__()

        self.p_train_imgs = sorted(glob(join('./Test_vis', f"*.bmp")))
        self.p_train_inf_imgs = sorted(glob(join('./Test_irs', f"*.bmp")))
        self.p_train_masks = sorted(glob(join("./tno_mask3", f"*.png")))
        self.p_train_vars = sorted(glob(join("./tno_vars", f"*.png")))
        self.p_crf_masks = sorted(glob(join("./tno_crf", f"*.png")))
        self.name = "tno"
        self.size = len(self.p_train_imgs)

    def __len__(self):
        return len(self.p_train_imgs)

    def __getitem__(self, index: int) -> dict:

        input_ir = imageio.imread(self.p_train_inf_imgs[index], as_gray=True)
        input_ir = np.reshape(input_ir, [1, input_ir.shape[0], input_ir.shape[1]])

        input_vi = imageio.imread(self.p_train_imgs[index], as_gray=True)
        input_vi = np.reshape(input_vi, [1, input_vi.shape[0], input_vi.shape[1]])

        image = input_vi
        inf_image = input_ir

        filename = self.p_train_imgs[index].split('/')[-1].split('.')[0]
        input_mk = imageio.imread(self.p_train_masks[index], as_gray=True)
        input_mk = np.reshape(input_mk, [1, input_mk.shape[0], input_mk.shape[1]])

        # 初始化vars和伪标签参数
        if len(self.p_train_vars) != 0:  # for warmup
            var = cv2.imread(self.p_train_vars[index], 0)
        else:
            var = np.zeros_like(input_mk)  # 初始化

        if len(self.p_crf_masks) != 0:  # for warmup
            crf = cv2.imread(self.p_crf_masks[index], 0)
        else:
            crf = np.zeros_like(input_mk)  # 初始化

        masks = torch.from_numpy(input_mk).float()

        image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        masks = torch.tensor(np.expand_dims(masks, 0)).float()
        masks = masks / 255.
        var = torch.from_numpy(np.expand_dims(np.stack(var, axis=0), 0)).unsqueeze(0).float()
        crf = torch.from_numpy(np.expand_dims(np.stack(crf, axis=0), 0)).unsqueeze(0).float()

        return filename, image, inf_image, masks, crf, var

    def save(self, name, pred):
        fnl_folder = './tno_mask/'
        pseudo = pred[0][0].cpu().detach().numpy()
        im_path = os.path.join(fnl_folder, name + '.png')
        imsave(im_path, pseudo)

    def update(self):
        self.p_train_masks = sorted(glob(join("./tno_mask", f"*.png")))

    def update_var(self):
        self.p_train_vars = sorted(glob(join("./tno_vars", f"*.png")))

    def update_crf(self):
        self.p_train_vars = sorted(glob(join("./tno_crf", f"*.png")))

    def selection(self, name, pred1, pred2):
        v_pred = pred1.cpu().detach().numpy()[0][0]
        i_pred = pred2.cpu().detach().numpy()[0][0]

        pred_as = np.stack([v_pred, i_pred], axis=0)  # n, h, w
        pred_as_var = np.var(pred_as, axis=0)  # h, w 得到方差图
        w, h = i_pred.shape
        pred_as_var = cv2.resize(pred_as_var, dsize=(h, w),
                                 interpolation=cv2.INTER_LINEAR)  # cv2 resize w, h,
        pred_as_var = np.round(pred_as_var*255)
        cv2.imwrite(os.path.join("./tno_vars", f'{name}.png'), pred_as_var)

    def generate(self, image, pred, name):
        pred = pred[0][0].cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
        std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
        orig_img = image[0].numpy().transpose(1, 2, 0)
        orig_img = ((orig_img * std + mean) * 127.5).astype(np.uint8)

        pred = (pred > 0.5).astype(np.uint8)
        pred = crf_inference_label(orig_img, pred)
        pred = cv2.medianBlur(pred.astype(np.uint8), 7)
        cv2.imwrite(os.path.join("./tno_crf", f'{name}.png'), pred)


class Train_RoadSceneDataset(data.Dataset):
    def __init__(
            self,
    ):
        """At this moment, only test set is considered."""
        super(Train_RoadSceneDataset, self).__init__()

        self.p_train_imgs = sorted(glob(join('./VIS', f"*.jpg")))
        self.p_train_inf_imgs = sorted(glob(join('./IF', f"*.jpg")))
        self.p_train_masks = sorted(glob(join("./roads_mask", f"*.png")))
        self.p_train_vars = sorted(glob(join("./road_vars", f"*.png")))
        self.p_crf_masks = sorted(glob(join("./road_crf", f"*.png")))
        self.name = "tno"
        self.size = len(self.p_train_imgs)

    def __len__(self):
        return len(self.p_train_imgs)

    def __getitem__(self, index: int) -> dict:

        input_vi = imageio.imread(self.p_train_imgs[index], as_gray=True)
        input_vi = np.reshape(input_vi, [1, input_vi.shape[0], input_vi.shape[1]])

        input_ir = imageio.imread(self.p_train_inf_imgs[index], as_gray=True)
        input_ir = np.reshape(input_ir, [1, input_ir.shape[0], input_ir.shape[1]])

        image = input_vi
        inf_image = input_ir

        filename = self.p_train_imgs[index].split('/')[-1].split('.')[0]
        input_mk = imageio.imread(self.p_train_masks[index], as_gray=True)
        input_mk = np.reshape(input_mk, [1, input_mk.shape[0], input_mk.shape[1]])

        # 初始化vars和伪标签参数
        if len(self.p_train_vars) != 0:  # for warmup
            var = cv2.imread(self.p_train_vars[index], 0)
        else:
            var = np.zeros_like(input_mk)  # 初始化

        if len(self.p_crf_masks) != 0:  # for warmup
            crf = cv2.imread(self.p_crf_masks[index], 0)
        else:
            crf = np.zeros_like(input_mk)  # 初始化

        masks = torch.from_numpy(input_mk)

        image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        masks = torch.tensor(np.expand_dims(masks, 0)).float()
        masks = masks / 255.
        var = torch.from_numpy(np.expand_dims(np.stack(var, axis=0), 0)).unsqueeze(0).float()
        crf = torch.from_numpy(np.expand_dims(np.stack(crf, axis=0), 0)).unsqueeze(0).float()

        return filename, image, inf_image, masks, crf, var

    def save(self, name, pred):
        fnl_folder = './road_mask/'
        pseudo = pred[0, 0].cpu().detach().numpy()
        im_path = os.path.join(fnl_folder, name + '.png')
        imsave(im_path, pseudo)

    def update(self):
        self.p_train_masks = sorted(glob(join("./road_mask", f"*.png")))

    def update_var(self):
        self.p_train_vars = sorted(glob(join("./road_vars", f"*.png")))

    def update_crf(self):
        self.p_train_vars = sorted(glob(join("./road_crf", f"*.png")))

    def selection(self, name, pred1, pred2):
        v_pred = pred1.cpu().detach().numpy()[0][0]
        i_pred = pred2.cpu().detach().numpy()[0][0]

        pred_as = np.stack([v_pred, i_pred], axis=0)  # n, h, w
        pred_as_var = np.var(pred_as, axis=0)  # h, w 得到方差图
        w, h = i_pred.shape
        pred_as_var = cv2.resize(pred_as_var, dsize=(h, w),
                                 interpolation=cv2.INTER_LINEAR)  # cv2 resize w, h,
        pred_as_var = np.round(pred_as_var*255)
        # Image.fromarray(pred_as_var*225.).convert('L').save(os.path.join("./road_vars", f'{name}.png'))
        cv2.imwrite(os.path.join("./road_vars", f'{name}.png'), pred_as_var)

    def generate(self, image, pred, name):
        pred = pred[0][0].cpu().detach().numpy().squeeze()

        pred = 255 * (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        orig_img = cv2.imread(os.path.join('./VIS/', "{}".format(name.split(".")[0] + ".jpg")), 1)
        crf_pred = crf_refine(
            orig_img,
            pred.astype('uint8'))
        Image.fromarray((crf_pred * 255)).convert('L').save(os.path.join("./road_crf", f'{name}.png'))

