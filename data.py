import os, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
from models.common import RGB2YCrCb, clamp
import imageio
import torchvision.transforms.functional as TF
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
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
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


def get_tnoloader(config):
    dataset = Train_TNODataset()
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader


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

        padding = 0
        input_ir = imread(self.p_train_inf_imgs[index]) / 127.5 - 1.0
        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([1, w, h])

        input_vi = imread(self.p_train_imgs[index]) / 127.5 - 1.0
        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([1, w, h])
        image = input_vi
        inf_image = input_ir

        filename = self.p_train_imgs[index].split('/')[-1].split('.')[0]
        input_mk = imread(self.p_train_masks[index]) / 127.5 - 1.0
        input_mk = np.lib.pad(input_mk, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_mk.shape
        input_mk = input_mk.reshape([1, w, h])

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

        image = TF.to_tensor(image).float()  # TF.normalize(TF.to_tensor(image), self.mean, self.std)  # 源代码
        image = clamp(image)
        image = image.permute(1, 2, 0)
        image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = TF.to_tensor(inf_image).float()
        inf_image = clamp(inf_image)
        inf_image = inf_image.permute(1, 2, 0)
        inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        masks = torch.tensor(np.expand_dims(masks, 0)).float()
        var = torch.from_numpy(np.expand_dims(np.stack(var, axis=0), 0)).unsqueeze(0).float()
        crf = torch.from_numpy(np.expand_dims(np.stack(crf, axis=0), 0)).unsqueeze(0).float()

        # # masks = np.asarray(masks, np.int64)
        # if masks.max() > 1.0:
        #     masks = masks > 0
        # masks = masks.permute(2, 1, 0)

        return filename, image, inf_image, masks, crf, var

    def save(self, name, pred):
        fnl_folder = './tno_mask/'
        pseudo = pred[0].cpu().detach().numpy()
        im_path = os.path.join(fnl_folder, name + '.png')
        imsave(im_path, pseudo[0])

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
        pred_as_var = np.round(pred_as_var * 255)
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


class Tno_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        # self.images = get_image_list(name, config, 'test')  # self.gts
        self.p_train_imgs = sorted(glob(join('./Test_vis', f"*.bmp")))
        self.p_train_inf_imgs = sorted(glob(join('./Test_irs', f"*.bmp")))
        # 按照数字进行排序后按顺序读取文件夹下的图片
        self.p_train_gts = sorted(list(paths.list_images("./Test_mask")))
        # 给读取图像排序遍历，数字由小到大
        self.p_train_gts.sort(key=lambda x: int((x.split('/')[-1]).split('.')[0]))

        self.size = len(self.p_train_imgs)
        self.dataset_name = name

    def load_data(self, index):
        padding = 0
        input_ir = imread(self.p_train_inf_imgs[index]) / 127.5 - 1.0
        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([1, w, h])

        input_vi = imread(self.p_train_imgs[index]) / 127.5 - 1.0
        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([1, w, h])

        image = input_vi
        inf_image = input_ir

        name = self.p_train_imgs[index].split('/')[-1].split('.')[0]

        input_mk = imread(self.p_train_gts[index]) / 127.5 - 1.0
        input_mk = np.lib.pad(input_mk, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_mk.shape
        input_mk = input_mk.reshape([w, h])
        masks = torch.from_numpy(input_mk)

        image = TF.to_tensor(image).float()  # TF.normalize(TF.to_tensor(image), self.mean, self.std)  # 源代码
        image = clamp(image)
        image = image.permute(1, 2, 0)
        image = torch.tensor(np.expand_dims(image, 0)).float()
        inf_image = TF.to_tensor(inf_image).float()
        inf_image = clamp(inf_image)
        inf_image = inf_image.permute(1, 2, 0)
        inf_image = torch.tensor(np.expand_dims(inf_image, 0)).float()

        # # masks = np.asarray(masks, np.int64)
        # if masks.max() > 1.0:
        #     masks = masks > 0
        # masks = masks.permute(2, 1, 0)

        return image, inf_image, masks, name


def test_data():
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset'}
    dataset = 'SOD'
    
    '''
    data_loader = Test_Dataset(dataset, config)
    #data_loader = Train_Dataset(dataset, config)
    data_size = data_loader.size
    
    for i in range(data_size):
        img, gt, name = data_loader.load_data(i)
        #img, gt = data_loader.__getitem__(i)
        new_img = (img * std + mean) * 255.
        #new_img = gt * 255
        print(np.min(new_img), np.max(new_img))
        new_img = (new_img).astype(np.uint8)
        #print(new_img.shape).astype(np.)
        im = Image.fromarray(new_img)
        #im.save('temp/' + name + '.jpg')
        im.save('temp/' + str(i) + '.jpg')
    
    '''
    
    data_loader = Val_Dataset(dataset, config)
    imgs, gts, names = data_loader.load_all_data()
    print(imgs.shape, gts.shape, len(names))


if __name__ == "__main__":
    test_data()