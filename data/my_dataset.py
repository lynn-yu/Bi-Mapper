import numpy as np

import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
Img_path = ['ImgLeft','ImgFront','Imgright','ImgBack']
def my_dataset(Dataset):
    def __init__(self, dataroot):
        super(my_dataset, self).__init__()
        self.dataroot = dataroot
        self.data_conf = {
        'image_size': [128, 352],
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }
        self.len = 100
    def __len__(self):
        return self.len
    def get_intrinsic(self,i):
        d = dict()
        d[0] = np.array([[273.7, 0.0, 334.9], [0, 273.4, 244.8], [0, 0, 1]])#左
        d[1] = np.array([[462.9, 0.0, 328.1], [0, 462.3, 183.2], [0, 0, 1]])#前
        d[2] = np.array([[273.2, 0.0, 328.7], [0, 274.1, 219.2], [0, 0, 1]])#右
        d[3] = np.array([[274.2, 0.0, 332.2], [0, 272.8, 239.1], [0, 0, 1]])#后
        return d[i]
    def get_rot(self,i):
        d = dict()
        d[0] = np.array([[0.99, -0.015, -0.00], [0.005, -0.004, 0.99], [-0.015, -0.99, -0.004]])
        d[1] = np.array([[0.0, 0.0, 1], [-1, 0, 0], [0, -1, 0]])
        d[2] = np.array([[-0.99, -0.0168, 0.0025], [-0.00223, -0.0160, -0.9998], [0.0168, -0.9997, 0.016]])
        d[3] = np.array([[-0.0027, -0.0307, -0.9995], [0.99992, -0.0117, -0.00237], [-0.0116, -0.9994, 0.0308]])
        return d[i]
    def get_trans(self,i):
        d = dict()
        d[0] = np.array([0, 0.263, 0.92])
        d[1] = np.array([0.35, 0.599, 0.92])
        d[2] = np.array([0, -0.263, 0.92])
        d[3] = np.array([ -0.365, 0.263, 0.92])
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
        for i in range(len(Img_path)):
            if i == 4:
              imgs.append(torch.zeros_like(img))
              a = np.array([[0,0,0],[0,0,0],[0,0,0]])
              b = np.array([0,0,0])
              trans.append(torch.Tensor(b))  # 标定位移变换，相机到车辆的变换
              rots.append(torch.Tensor(a))  # 标定旋转变换
              intrins.append(torch.Tensor(a))
              feas.append(torch.zeros_like(fea))
            root = os.path.join(self.path,Img_path[i])
            n = str(id)
            n = n.zfill(6)
            root = root + n + ".jpg"
            img = Image.open(root)
            resize, resize_dims = self.sample_augmentation()  # 图片变换尺度，变换大小，目标数据尺寸
            img_o_res, _, _ = img_transform(img, resize, resize_dims)  # 输出3*3旋转和平移矩阵
            img_o = nor_img(img)  # 3 900 1600
            img_origin.append(nor_img(img_o_res))
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)  # 输出3*3旋转和平移矩阵
            img = normalize_img(img)  # 归一化3 128 325
            imgs.append(img)  # 不同角度相机累积

            trans.append(torch.Tensor(get_trans(i)))  # 标定位移变换，相机到车辆的变换
            rots.append(torch.Tensor(get_rot(i)))  # 标定旋转变换
            intrins.append(torch.Tensor(get_intrinsic(i)))

            fea = get_bev_cam(img_o,torch.Tensor(get_intrinsic(i)) , 1)  # ,fea_1
            fea_p = denor_img(fea)
            fea_p = fea_p.resize(resize_dims)
            fea = normalize_img(fea_p)
            # fea= nor_img(fea_p)#显示原图
            feas.append(fea)
        imgs.append(torch.zeros_like(img))
        a = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        b = np.array([0, 0, 0])
        trans.append(torch.Tensor(b))  # 标定位移变换，相机到车辆的变换
        rots.append(torch.Tensor(a))  # 标定旋转变换
        intrins.append(torch.Tensor(a))
        feas.append(torch.zeros_like(fea))
        return torch.stack(img_origin),torch.stack(feas),torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), None, None
    def __getitem__(self, idx):
        img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(idx)
        return img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots, None, None, None, None