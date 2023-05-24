import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        else:
            raise NotImplementedError


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))#内部含有sigmod

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


class DiscriminativeLoss(nn.Module):
    def __init__(self, embed_dim, delta_v, delta_d):
        super(DiscriminativeLoss, self).__init__()
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(self, embedding, seg_gt):
        if embedding is None:
            return 0, 0, 0
        bs = embedding.shape[0]#4

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(bs):
            embedding_b = embedding[b]  # (embed_dim, H, W) 16 200 400 预测16个对象位置
            seg_gt_b = seg_gt[b]# 200 400

            labels = torch.unique(seg_gt_b)#挑出所有不重复数字1 2  ...，所有对象
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:#每个类别数字
                seg_mask_i = (seg_gt_b == lane_idx)#取出其中一个类别的数字
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / bs
        dist_loss = dist_loss / bs
        reg_loss = reg_loss / bs
        return var_loss, dist_loss, reg_loss


def calc_loss():
    pass

xmin, xmax = -5, 5#-30 30
xx_l = xmax- xmin
num_x = 32
zmin, zmax = 3, 29
zz_l = zmax - zmin
num_z = 88
#h = 0.5
def plane_grid_cam(trans, rots,h):

    B,N,_ = trans.size()#4 6  3


    # y = torch.linspace(xmin, xmax, num_x, dtype=torch.double).cuda()
    # x = torch.linspace(ymin, ymax, num_y, dtype=torch.double).cuda()
    x = torch.linspace(xmin, xmax, num_x)
    z = torch.linspace(zmin, zmax, num_z)

    x = x.cuda()
    z = z.cuda()

    x, z = torch.meshgrid(x, z)

    x = x.flatten()#1 3200
    z = z.flatten()
    x = x.unsqueeze(0).repeat(B*N, 1)#24 3200
    z = z.unsqueeze(0).repeat(B*N, 1)

    y = torch.ones_like(x)
    y= y*h
    d = torch.ones_like(x)
    y = y.cuda()
    d = d.cuda()

    coords = torch.stack([x, y, z,d], axis=1)# 24 3 3200
    Rs = torch.eye(4, device=rots.device).repeat(B, N, 1, 1)#转为车辆坐标系
    Rs[:, :, :3, :3] = rots
    Ts = torch.eye(4, device=trans.device).repeat(B, N, 1, 1)
    Ts[:, :, :3, 3] = trans
    RTs = Rs @ Ts#b 6 4 4
    RTs = RTs.view(-1,4,4)
    # rots = rots.view(-1,3,3)
    # ego_coords = rots @ coords#24 4 4 * 24 4 3200 = 24 4 3200
    # trans = trans.view(-1,3,1)
    # ego_coords += trans
    #RTs = RTs.cuda()
    ego_coords = RTs @ coords
    ego_coords = torch.stack([ego_coords[:, 0], ego_coords[:, 1]], axis=1)# 24 2 3200
    ego_coords = ego_coords.view(B*N,2,num_x,num_z)
    ego_coords = ego_coords.view(B, N, 2, num_x, num_z)
    ego_coords = ego_coords.permute(0, 1, 3, 4, 2).contiguous()#4 6 40 80 2
    return ego_coords

def get_cam_gt(img,ego_coords):#传入真值
    _,  img_c , img_h, img_w= img.shape#4 5 200 400
    img = img.permute(0,2,3,1).contiguous()#4 200 400 5
    B, N, pix_h, pix_w, pix_c = ego_coords.shape#4 6 40 80 2
    out_shape = (B, N, pix_h, pix_w, img_c)

    pix_x, pix_y = torch.split(ego_coords, 1, dim=-1)  # [B, n,pix_h, pix_w,1]
    pix_x = pix_x / 0.15 + 200
    pix_y = pix_y / 0.15 + 100
    # Rounding
    pix_x0 = torch.floor(pix_x)
    pix_x1 = pix_x0 + 1
    pix_y0 = torch.floor(pix_y)
    pix_y1 = pix_y0 + 1

    # Clip within image boundary
    y_max = (img_h - 1)
    x_max = (img_w - 1)

    pix_x0 = torch.clip(pix_x0, 0, x_max)
    pix_y0 = torch.clip(pix_y0, 0, y_max)
    pix_x1 = torch.clip(pix_x1, 0, x_max)
    pix_y1 = torch.clip(pix_y1, 0, y_max)  #

    # Weights [B, n,pix_h, pix_w, 1]
    wt_x0 = pix_x1 - pix_x
    wt_x1 = pix_x - pix_x0
    wt_y0 = pix_y1 - pix_y
    wt_y1 = pix_y - pix_y0

    # indices in the image to sample from
    dim = img_w

    # Apply the lower and upper bound pix coord
    base_y0 = pix_y0 * dim
    base_y1 = pix_y1 * dim

    # 4 corner vert ices
    idx00 = (pix_x0 + base_y0).view(B,N, -1, 1).repeat(1, 1, 1, img_c).long()
    idx01 = (pix_x0 + base_y1).view(B,N, -1, 1).repeat(1, 1, 1, img_c).long()
    idx10 = (pix_x1 + base_y0).view(B,N, -1, 1).repeat(1, 1, 1, img_c).long()
    idx11 = (pix_x1 + base_y1).view(B,N, -1, 1).repeat(1, 1, 1, img_c).long()

    # Gather pixels from image using vertices

    imgs_flat = img.reshape([B, -1, img_c]).unsqueeze(1)
    imgs_flat = imgs_flat.repeat(1,N,1,1)
    #imgs_flat = imgs_flat.cuda()
    im00 = torch.gather(imgs_flat, 2, idx00).reshape(out_shape)
    im01 = torch.gather(imgs_flat, 2, idx01).reshape(out_shape)
    im10 = torch.gather(imgs_flat, 2, idx10).reshape(out_shape)
    im11 = torch.gather(imgs_flat, 2, idx11).reshape(out_shape)

    # Apply weights [pix_h, pix_w, 1]
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    gt = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return gt#b n h w 5


def view_loss(gt,trans,rots,h=0.5):

    ego_coords = plane_grid_cam(trans, rots,h)#4 6 2 40 80
    cam_gt = get_cam_gt(gt,ego_coords)#b n h w 5
    cam_gt = cam_gt.permute(0,1,4,2,3).contiguous()
    b,n,c,h,w = cam_gt.size()
    cam_gt = cam_gt.view(-1, c, h, w)
    return cam_gt









