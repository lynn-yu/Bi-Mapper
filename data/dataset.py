import os
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from data.rasterize import preprocess_map
from data.const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
from data.vector_map import VectorizedLocalMap
from data.lidar import get_lidar_data
from data.image import normalize_img, img_transform,nor_img,denor_img,denormalize_img,img_transform_rot
from data.utils import label_onehot_encoding,eulerAngles2rotationMat,get_files_in_folder,one_hot_encode_image,parse_convert_xml
from nuscenes.utils.data_classes import Box
from data.utils import gen_dx_bx
from torch.nn import functional as F
import matplotlib.pyplot as plt

STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

Img_path = ['ImgLeft','ImgFront','ImgRight','ImgBack']

Img_path_four = ['left','front','right','rear']

def to_image(x):
    return (255 * x).byte().cpu().numpy().transpose(1, 2, 0)
class mydataset(Dataset):
    def __init__(self, dataroot):
        super(mydataset, self).__init__()
        self.path = dataroot
        self.data_conf = {
        'image_size': [128, 352],
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }
        self.len = 200
    def __len__(self):
        return self.len
    def get_intrinsic(self,i):
        d = dict()
        d[0] = np.array([[273.7, 0.0, 334.9], [0, 273.4, 244.8], [0, 0, 1]])#左
        d[1] = np.array([[462.9, 0.0, 328.1], [0, 462.3, 183.2], [0, 0, 1]])#前
        d[2] = np.array([[273.2, 0.0, 328.7], [0, 274.1, 219.2], [0, 0, 1]])#右
        #d[3] = np.array([[273.7, 0.0, 334.9], [0, 273.4, 244.8], [0, 0, 1]])#左
        d[3] = np.array([[274.2, 0.0, 332.2], [0, 272.8, 239.1], [0, 0, 1]])#后
        #d[5] = np.array([[273.2, 0.0, 328.7], [0, 274.1, 219.2], [0, 0, 1]])  # 右
        return d[i]
    def get_rot(self,i):
        d = dict()
        d[0] = np.array([[1.0, -0.015, -0.0055], [0.005, -0.0048, 1.0], [-0.015, -1.0, -0.004]])
        d[1] = np.array([[0.0, 0.0, 1], [-1, 0, 0], [0, -1, 0]])
        d[2] = np.array([[-0.999, -0.0168, 0.00257], [-0.0023, -0.0160, -0.9999], [0.0168, -0.9997, 0.016]])
        #d[3] = np.array([[1.0, -0.015, -0.0055], [0.005, -0.0048, 1.0], [-0.015, -1.0, -0.004]])
        d[3] = np.array([[-0.0027, -0.0307, -0.9995], [0.99992, -0.0117, -0.00237], [-0.0116, -0.9994, 0.0308]])
        #d[5] = np.array([[-0.999, -0.0168, 0.00257], [-0.0023, -0.0160, -0.9999], [0.0168, -0.9997, 0.016]])
        return d[i]
    def get_trans(self,i):
        d = dict()
        d[0] = np.array([0, 0.263, 0.92])
        d[1] = np.array([0.35, 0.599, 0.92])
        d[2] = np.array([0, -0.263, 0.92])
        #d[3] = np.array([0, 0.263, 0.92])
        d[3] = np.array([ -0.365, 0.263, 0.92])
        #d[5] = np.array([0, -0.263, 0.92])
        return d[i]
    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size']  # 128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)  # （1，1）变换尺度
        resize_dims = (fW, fH)  # （128,352）目标数据尺寸
        return resize, resize_dims
    def get_img(self,id):
        imgs = []
        img_origin = []
        feas = []
        trans = []
        rots = []
        intrins = []
        for i in range(len(Img_path)):
            # if i == 3:
            #   imgs.append(torch.zeros_like(img))
            #   img_origin.append(torch.zeros_like(img))
            #   a = np.array([[0,0,0],[0,0,0],[0,0,0]])
            #   b = np.array([0,0,0])
            #   trans.append(torch.Tensor(b))  # 标定位移变换，相机到车辆的变换
            #   rots.append(torch.Tensor(a))  # 标定旋转变换
            #   intrins.append(torch.Tensor(a))
            #   feas.append(torch.zeros_like(fea))
            root = os.path.join(self.path,Img_path[i])
            n = str(id)
            n = n.zfill(6)
            root = root + '/' + n + ".jpg"
            img = Image.open(root)
            resize, resize_dims = self.sample_augmentation()  # 图片变换尺度，变换大小，目标数据尺寸
            img_o_res, _, _ = img_transform(img, resize, resize_dims)  # 输出3*3旋转和平移矩阵
            img_o = nor_img(img)  # 3 900 1600
            img_origin.append(nor_img(img_o_res))
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)  # 输出3*3旋转和平移矩阵
            img = normalize_img(img)  # 归一化3 128 325
            imgs.append(img)  # 不同角度相机累积

            trans.append(torch.Tensor(self.get_trans(i)))  # 标定位移变换，相机到车辆的变换
            rots.append(torch.Tensor(self.get_rot(i)))  # 标定旋转变换
            intrins.append(torch.Tensor(self.get_intrinsic(i)))

            fea = get_bev_cam(img_o,torch.Tensor(self.get_intrinsic(i)) , 1)  # ,fea_1
            fea_p = denor_img(fea)
            fea_p = fea_p.resize(resize_dims)
            fea = normalize_img(fea_p)
            #fea= nor_img(fea_p)#显示原图
            feas.append(fea)
        # imgs.append(torch.zeros_like(img))
        # a = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # b = np.array([0, 0, 0])
        # trans.append(torch.Tensor(b))  # 标定位移变换，相机到车辆的变换
        # rots.append(torch.Tensor(a))  # 标定旋转变换
        # intrins.append(torch.Tensor(a))
        # feas.append(torch.zeros_like(fea))
        return torch.stack(img_origin),torch.stack(feas),torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.tensor([0]), torch.tensor([0])
    def __getitem__(self, idx):
        img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots = self.get_img(idx)
        return img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots, torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
def my_semantic_dataset( dataroot, bsz, nworkers):
    train_dataset = mydataset(dataroot)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers, drop_last=True)

    return train_loader

class vir_dataset(Dataset):
    def __init__(self, dataroot,is_train=False):
        super(vir_dataset, self).__init__()
        self.root = dataroot
        self.data_conf = {
        'image_size': [128, 352],
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }
        self.is_train = is_train

        if self.is_train:
            self.path = os.path.join(self.root,"train")
        else:
            self.path = os.path.join(self.root, "val")
        self.example,self.lable = self.get_examples()
    def __len__(self):
        if self.is_train:
            self.len = 33199
            return self.len
        else:
            self.len = 3731
            return self.len

    def get_intrinsic(self,i):
        d = dict()
        d[0] = np.array([[278.283, 0.0, 482.0], [0, 408.1295, 302.0], [0, 0, 1]])#左
        d[1] = np.array([[278.283, 0.0, 482.0], [0, 408.1295, 302.0], [0, 0, 1]])#前
        d[2] = np.array([[278.283, 0.0, 482.0], [0, 408.1295, 302.0], [0, 0, 1]])#右
        #d[3] = np.array([[278.283, 0.0, 482.0], [0, 408.1295, 302.0], [0, 0, 1]])  # 左
        d[3] = np.array([[278.283, 0.0, 482.0], [0, 408.1295, 302.0], [0, 0, 1]])#后
        #d[5] = np.array([[278.283, 0.0, 482.0], [0, 408.1295, 302.0], [0, 0, 1]])  # 右
        return d[i]
    def get_rot(self,i):
        d = dict()
        d[0] = eulerAngles2rotationMat([0,0,90])
        d[1] = eulerAngles2rotationMat([0,0,0])
        d[2] = eulerAngles2rotationMat([0,0,-90])
        #d[3] = eulerAngles2rotationMat([0, 0, 90])
        d[3] = eulerAngles2rotationMat([0,0,180])
        #d[5] = eulerAngles2rotationMat([0, 0, -90])
        return d[i]
    def get_trans(self,i):
        d = dict()
        d[0] = np.array([0.5, 0.5, 1.5])
        d[1] = np.array([1.7, 0, 1.4])
        d[2] = np.array([0.5, -0.5, 1.5])
        #d[3] = np.array([0.5, 0.5, 1.5])
        d[3] = np.array([ -0.6, 0, 1.4])
        #d[5] = np.array([0.5, -0.5, 1.5])
        return d[i]

    def get_examples(self):
        examples = dict()
        for i in range(len(Img_path_four)):
            root = os.path.join(self.path,Img_path_four[i])
            files = get_files_in_folder(root)
            examples[Img_path_four[i]] = files
        root_l = os.path.join(self.path, "bev")
        lable = get_files_in_folder(root_l)
        return examples,lable

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size']  # 128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)  # （1，1）变换尺度
        resize_dims = (fW, fH)  # （128,352）目标数据尺寸
        return resize, resize_dims
    def get_img(self,id):
        imgs = []
        img_origin = []
        feas = []
        trans = []
        rots = []
        intrins = []
        for i in range(len(Img_path_four)):
            root = self.example[Img_path_four[i]][id]
            img = Image.open(root)
            resize, resize_dims = self.sample_augmentation()  # 图片变换尺度，变换大小，目标数据尺寸
            img_o_res, _, _ = img_transform(img, resize, resize_dims)  # 输出3*3旋转和平移矩阵
            img_o = nor_img(img)  # 3 900 1600
            img_origin.append(nor_img(img_o_res))
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)  # 输出3*3旋转和平移矩阵
            img = normalize_img(img)  # 归一化3 128 325
            imgs.append(img)  # 不同角度相机累积

            trans.append(torch.Tensor(self.get_trans(i)))  # 标定位移变换，相机到车辆的变换
            rots.append(torch.Tensor(self.get_rot(i)))  # 标定旋转变换
            intrins.append(torch.Tensor(self.get_intrinsic(i)))

            fea = get_bev_cam(img_o,torch.Tensor(self.get_intrinsic(i)) , 1)  # ,fea_1
            fea_p = denor_img(fea)
            fea_p = fea_p.resize(resize_dims)
            fea = normalize_img(fea_p)
            #fea= nor_img(fea_p)#显示原图
            feas.append(fea)

        return torch.stack(img_origin),torch.stack(feas),torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.tensor([0]), torch.tensor([0])

    def get_label(self, idx):
        lable_root = self.lable[idx]
        lable = Image.open(lable_root)
        lable = lable.resize((400,200))
        lable = np.array(lable)
        #lable = lable[ 202:402,282:682, :]
        root = os.path.join(self.root,"one_hot_conversion/convert_3.xml")
        one_hot_palette_label = parse_convert_xml(root)
        one_hot_lable = one_hot_encode_image(lable,one_hot_palette_label)#
        #one_hot_lable = np.resize(one_hot_lable,(200,400,3))
        return one_hot_lable

    def __getitem__(self, idx):
        img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots = self.get_img(idx)
        semantic = self.get_label(idx)
        semantic = torch.tensor(semantic)
        semantic = semantic.permute(2,0,1).contiguous()
        semantic_mask = semantic!=0
        semantic_mask = torch.cat([(~torch.any(semantic_mask, axis=0)).unsqueeze(0), semantic_mask])

        return img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots, torch.tensor([0]), torch.tensor([0]), semantic_mask, torch.tensor([0])
def vir_semantic_dataset( version, root, data_conf, bsz, nworkers):
    dataroot = "/data/lsy/dataset/vir_road_dataset"
    train_dataset = vir_dataset(dataroot,is_train = True)
    val_dataset = vir_dataset(dataroot,is_train = False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader, val_loader

def plane_grid(xmin, xmax, num_x, zmin, zmax, num_z, intris,hw):
    h, w = intris.shape  # 4 6  3

    # y = torch.linspace(xmin, xmax, num_x, dtype=torch.double).cuda()
    # x = torch.linspace(ymin, ymax, num_y, dtype=torch.double).cuda()
    x = torch.linspace(xmin, xmax, num_x)
    z = torch.linspace(zmin, zmax, num_z)



    x, z = torch.meshgrid(x, z)

    x = x.flatten()  # 1 3200
    z = z.flatten()
    y = torch.ones_like(x)
    y = y*hw
    coords = torch.stack([x, y, z], axis=0)  # 3 xx

    ego_coords = intris @ coords  # 24 3 3 * 24 3 xx = 24 3 3200
    z = coords[2, :].unsqueeze(0).repeat(3, 1)
    ego_coords = ego_coords / z
    ego_coords = torch.stack([ego_coords[0], ego_coords[1]], axis=0)  # 24 2 3200
    ego_coords = ego_coords.view(2, num_x, num_z)
    ego_coords = ego_coords.permute(1,2,0).contiguous()  # 40 80 2
    return ego_coords

def get_origin_img_fea(img, ego_coords):
    img_c, img_h, img_w  = img.shape  #  3 900 1600
    img = img.permute(1,2,0).contiguous()#900 1600 3
    pix_h, pix_w, pix_c = ego_coords.shape  # 300 600 2
    out_shape = (pix_h, pix_w, img_c)

    pix_x, pix_y = torch.split(ego_coords, 1, dim=-1)  # [pix_h, pix_w,1]

    # Rounding
    pix_x0 = torch.floor(pix_x)
    pix_x1 = pix_x0 + 1
    pix_x2 = pix_x0 - 1
    pix_y0 = torch.floor(pix_y)
    pix_y1 = pix_y0 + 1
    pix_y2 = pix_y0 - 1

    # Clip within image boundary
    y_max = (img_h - 1)
    x_max = (img_w - 1)

    pix_x0 = torch.clip(pix_x0, 0, x_max)
    pix_y0 = torch.clip(pix_y0, 0, y_max)
    pix_x1 = torch.clip(pix_x1, 0, x_max)
    pix_y1 = torch.clip(pix_y1, 0, y_max)  #
    pix_x2 = torch.clip(pix_x2, 0, x_max)
    pix_y2 = torch.clip(pix_y2, 0, y_max)

    # Weights [B, n,pix_h, pix_w, 1]
    wt_x0 = pix_x1 - pix_x
    wt_x1 = pix_x - pix_x0
    wt_x2 = pix_x - pix_x2
    wt_y0 = pix_y1 - pix_y
    wt_y1 = pix_y - pix_y0
    wt_y2 = pix_y - pix_y2

    # indices in the image to sample from
    dim = img_w

    # Apply the lower and upper bound pix coord
    base_y0 = pix_y0 * dim
    base_y1 = pix_y1 * dim
    base_y2 = pix_y2 * dim
    # 4 corner vert ices
    idx00 = (pix_x0 + base_y0).view(-1, 1).repeat( 1, img_c).long()
    idx01 = (pix_x0 + base_y1).view(-1, 1).repeat( 1, img_c).long()
    idx02 = (pix_x0 + base_y2).view(-1, 1).repeat(1, img_c).long()
    idx10 = (pix_x1 + base_y0).view(-1, 1).repeat( 1, img_c).long()
    idx11 = (pix_x1 + base_y1).view(-1, 1).repeat(1, img_c).long()
    idx12 = (pix_x1 + base_y2).view(-1, 1).repeat(1, img_c).long()
    idx20 = (pix_x2 + base_y0).view(-1, 1).repeat(1, img_c).long()
    idx21 = (pix_x2 + base_y1).view(-1, 1).repeat(1, img_c).long()
    idx22 = (pix_x2 + base_y2).view(-1, 1).repeat(1, img_c).long()
    # Gather pixels from image using vertices

    imgs_flat = img.reshape([ -1, img_c])
    # imgs_flat = imgs_flat.repeat(1, N, 1, 1)

    im00 = torch.gather(imgs_flat, 0, idx00).reshape(out_shape)
    im01 = torch.gather(imgs_flat, 0, idx01).reshape(out_shape)
    im02 = torch.gather(imgs_flat, 0, idx02).reshape(out_shape)
    im10 = torch.gather(imgs_flat, 0, idx10).reshape(out_shape)
    im11 = torch.gather(imgs_flat, 0, idx11).reshape(out_shape)
    im12 = torch.gather(imgs_flat, 0, idx12).reshape(out_shape)
    im20 = torch.gather(imgs_flat, 0, idx20).reshape(out_shape)
    im21 = torch.gather(imgs_flat, 0, idx21).reshape(out_shape)
    im22 = torch.gather(imgs_flat, 0, idx22).reshape(out_shape)
    # Apply weights [pix_h, pix_w, 1]
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w02 = wt_x0 * wt_y2
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    w12 = wt_x1 * wt_y2
    w20 = wt_x2 * wt_y0
    w21 = wt_x2 * wt_y1
    w22 = wt_x2 * wt_y2
    #gt = w00 * im00 + w01 * im01 + w02 * im02+ w10 * im10 + w11 * im11 + w12 * im12 + w20 * im20 + w21 * im21 + w22 * im22 # 300 600 3
    gt = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    gt = gt.permute(2,0,1).contiguous()#3 300 600
    #gt_crop = gt[:,100:300,200:400]
    return gt

def get_bev_cam(img,intrins,h):
    xmin, xmax = -5, 5
    num_x = 400
    zmin, zmax = 3,29#2, 20
    num_z = 800

    plane = plane_grid(xmin, xmax, num_x, zmin, zmax, num_z, intrins,h)#
    plane_fea = get_origin_img_fea(img, plane)  #  3 400 800
    #plane_1 = plane_grid(xmin, xmax, num_x, zmin, zmax, num_z, intrins, 1)  # 400 800 2
    #plane_fea_1 = get_origin_img_fea(img, plane_1)  # 3 128 352
    return plane_fea#,plane_fea_1

class HDMapNetDataset(Dataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(HDMapNetDataset, self).__init__()
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]#patch大小 15 -15 =30
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]#patch大小 30 -30 =60
        canvas_h = int(patch_h / data_conf['ybound'][2])#网格化？200
        canvas_w = int(patch_w / data_conf['xbound'][2])# 400
        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.bx_cam = [0,-16]
        self.dx_cam = [0.23,0.5]
        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)#30 60
        self.canvas_size = (canvas_h, canvas_w)#200 400
        self.patch_size_cam = (12, 60)
        self.canvas_size_cam = (40, 200)  # 200 400
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)
        #self.vector_map_cam = VectorizedLocalMap(dataroot, patch_size=self.patch_size_cam, canvas_size=self.canvas_size_cam)
        self.vector_map_merge = VectorizedLocalMap(dataroot, patch_size=(30,75),canvas_size=(200,500))
        self.scenes = self.get_scenes(version, is_train)
        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

        return create_splits_scenes()[split]

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples#


    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        car_trans = ego_pose['translation']#车辆到世界坐标系的变换
        pos_rotation = Quaternion(ego_pose['rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size_i']#128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)#（1，1）变换尺度
        resize_dims = (fW, fH)#（128,352）目标数据尺寸
        return resize, resize_dims

    def sample_augmentation_f(self):
        fH, fW = self.data_conf['image_size']  # 128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)  # （1，1）变换尺度
        resize_dims = (fW, fH)  # （128,352）目标数据尺寸
        return resize, resize_dims

    def sample_augmentation_rot(self):
        self.data_conf['resize_lim'] = (0.193, 0.225)
        self.data_conf['bot_pct_lim'] = (0.0, 0.22)
        self.data_conf['rand_flip'] = True
        self.data_conf['rot_lim'] = (-5.4, -5.4)

        fH, fW = self.data_conf['image_size']
        if self.is_train:
            resize = np.random.uniform(*self.data_conf['resize_lim'])
            resize_dims = (int(IMG_ORIGIN_W*resize), int(IMG_ORIGIN_H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_conf['rot_lim'])
        else:
            resize = max(fH/IMG_ORIGIN_H, fW/IMG_ORIGIN_W)
            resize_dims = (int(IMG_ORIGIN_W*resize), int(IMG_ORIGIN_H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate



    def get_imgs(self, rec,h):#rec不同时间帧数据
        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []
        img_origin = []
        feas = []
        #resize, resize_dims, crop, flip, rotate = self.sample_augmentation_rot()
        for cam in CAMS:#不同相机位置名字
            samp = self.nusc.get('sample_data', rec['data'][cam])#获得本相机的图片数据
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])#图片存放路径
            resize_i, resize_dims_i = self.sample_augmentation()  # 图片变换尺度，变换大小，目标数据尺寸
            img = Image.open(imgname)
            img_o_res, _, _ = img_transform(img, resize_i, resize_dims_i)  # 输出3*3旋转和平移矩阵,resize图片大小
            img_o = nor_img(img)#3 900 1600

            img_origin.append(nor_img(img_o_res))


            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])  # 传感器标定数据
            trans.append(torch.Tensor(sens['translation']))  # 标定位移变换，相机到车辆的变换
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))  # 标定旋转变换
            intrins.append(torch.Tensor(sens['camera_intrinsic']))

            fea = get_bev_cam(img_o, torch.Tensor(sens['camera_intrinsic']),h)#,fea_1
            fea_p = denor_img(fea)
            fea_p = fea_p.resize(resize_dims_i)
            fea = normalize_img(fea_p)
            #fea= nor_img(fea_p)#显示原图
            feas.append(fea)

            #resize_f, resize_dims_f = self.sample_augmentation_f()  # 图片变换尺度，变换大小，目标数据尺寸
            img, post_rot, post_tran = img_transform(img, resize_i, resize_dims_i)#输出3*3旋转和平移矩阵

            #img, post_rot, post_tran = img_transform_rot(img, resize, resize_dims, crop, flip, rotate)

            img = normalize_img(img)#归一化3 128 325
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            imgs.append(img)#不同角度相机累积

        return torch.stack(img_origin),torch.stack(feas),torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)



    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']#此场景日志信息，采集地名字
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])#车本身在世界坐标系位姿
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])

        return vectors

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_origin,feas,imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec,1)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        vectors = self.get_vectors(rec)

        return img_origin,feas,imgs, trans, rots, intrins, post_trans, post_rots, car_trans, yaw_pitch_roll, vectors



class HDMapNetSemanticDataset_all(HDMapNetDataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(HDMapNetSemanticDataset_all, self).__init__(version, dataroot, data_conf, is_train)
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']

    def get_semantic_map(self, rec):
        vectors = self.get_vectors(rec)
        instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, 3, self.thickness, self.angle_class)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])

        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks
        direction_masks = direction_masks / direction_masks.sum(0)
        return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks #semantic_masks_cam,

    def get_semantic_map_merge(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']  # 此场景日志信息，采集地名字
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])  # 车本身在世界坐标系位姿
        vectors = self.vector_map_merge.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        instance_masks, forward_masks, backward_masks = preprocess_map(vectors,(30,75),(200,500), 3,self.thickness, self.angle_class)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        return  semantic_masks
    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_origin,feas,imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec,1)

        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        semantic_masks, instance_masks, _, _, direction_masks= self.get_semantic_map(rec) #  6 4 32 88
        return img_origin,feas,imgs, trans, rots, intrins, post_trans, post_rots,car_trans, yaw_pitch_roll, semantic_masks,  instance_masks


def semantic_dataset(version, dataroot, data_conf, bsz, nworkers):
    train_dataset = HDMapNetSemanticDataset_all(version, dataroot, data_conf, is_train=True)
    val_dataset = HDMapNetSemanticDataset_all(version, dataroot, data_conf, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader, val_loader


if __name__ == '__main__':
    data_conf = {
        'image_size': [128, 352],
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = vir_dataset("/data/lsy/dataset/vir_road_dataset")
    for idx in range(5):#数据集大小
        #获得这id的相关数据
        img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots, car_trans, yaw_pitch_roll, semantic,_ = dataset.__getitem__(idx)
        # for i in range(4):
        #     plt.imshow(to_image(img_origin[i]))
        #     plt.show()
        plt.figure(figsize=(4, 2), dpi=100)
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.imshow(semantic[1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.imshow(semantic[2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        plt.imshow(semantic[3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        plt.show()



