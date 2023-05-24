import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from loss import SimpleLoss, DiscriminativeLoss, view_loss,FocalLoss
from data.dataset import HDMapNetSemanticDataset_all, CAMS,HDMapNetSemanticDataset_cam,HDMapNetDataset
from data.utils import get_proj_mat, perspective
from data.image import denormalize_img
from data.viz_data import BaseViz
import cv2
def to_image(x):
    return (255 * x).byte().cpu().numpy().transpose(1, 2, 0)
def vis_label(dataroot, version, xbound, ybound):
    data_conf = {
        'image_size': (900, 1600),
        'xbound': xbound,
        'ybound': ybound,
        'zbound': args.zbound,
        'dbound': args.dbound
    }

    color_map = np.random.randint(0, 256, (256, 3))
    color_map[0] = np.array([0, 0, 0])
    colors_plt = ['r', 'b', 'g']

    dataset = HDMapNetDataset(version=version, dataroot=dataroot, data_conf=data_conf, is_train=False)
    gt_path = os.path.join(dataroot, 'samples', 'GT')
    if not os.path.exists(gt_path):
        os.mkdir(gt_path)

    car_img = Image.open('icon/car.png')#车辆模型
    for idx in tqdm.tqdm(range(dataset.__len__())):
        rec = dataset.nusc.sample[idx]#不同时间帧sample
        img_origin,feas,imgs, trans, rots, intrins, post_trans, post_rots = dataset.get_imgs(rec,1)#不同相机的数据
        vectors = dataset.get_vectors(rec)

        lidar_top_path = dataset.nusc.get_sample_data_path(rec['data']['LIDAR_TOP'])

        base_path = lidar_top_path.split('/')[-1].replace('__LIDAR_TOP__', '_').split('.')[0]
        base_path = os.path.join(gt_path, base_path)

        if not os.path.exists(base_path):
            os.mkdir(base_path)

        plt.figure(figsize=(4, 2))
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        plt.axis('off')
        for vector in vectors:
            pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
            pts = pts[:pts_num]
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[line_type])

        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

        map_path = os.path.join(base_path, 'MAP.png')
        plt.savefig(map_path, bbox_inches='tight', dpi=400)
        plt.close()

        for img, intrin, rot, tran, cam in zip(imgs, intrins, rots, trans, CAMS):
            img = denormalize_img(img)
            P = get_proj_mat(intrin, rot, tran)
            plt.figure(figsize=(9, 16))
            fig = plt.imshow(img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.xlim(1600, 0)
            plt.ylim(900, 0)
            plt.axis('off')
            for vector in vectors:
                pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                pts = pts[:pts_num]
                zeros = np.zeros((pts_num, 1))
                ones = np.ones((pts_num, 1))
                world_coords = np.concatenate([pts, zeros, ones], axis=1).transpose(1, 0)
                pix_coords = perspective(world_coords, P)
                x = np.array([pts[0] for pts in pix_coords])
                y = np.array([pts[1] for pts in pix_coords])
                plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy',
                           angles='xy', scale=1, color=colors_plt[line_type])

            cam_path = os.path.join(base_path, f'{cam}.png')
            plt.savefig(cam_path, bbox_inches='tight', pad_inches=0, dpi=400)
            plt.close()

def vis_gt(dataroot, version, xbound, ybound):
    data_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': args.zbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'image_size': args.image_size,
        'dbound': args.dbound
    }
    dataset = HDMapNetSemanticDataset_all(version=version, dataroot=dataroot, data_conf=data_conf, is_train=False)
    for idx in tqdm.tqdm(range(dataset.__len__())):#dataset.__len__()
        rec = dataset.nusc.sample[idx]#不同时间帧sample
        # img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots = dataset.get_imgs(rec, 1)
        # for j in range(6):
        #     plt.imshow(to_image(img_origin[j]))
        #     plt.show()
        semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks = dataset.get_semantic_map(rec)

        # plt.imshow(semantic_masks[1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        # plt.imshow(semantic_masks[2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        # plt.imshow(semantic_masks[3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        # plt.show()
        img_origin,feas,imgs, trans, rots, intrins, post_trans, post_rots = dataset.get_imgs(rec, 1)
        #cam_to_world_t, cam_to_world_r =dataset.get_cam_pose(rec, trans.numpy(), rots.numpy())
        #semantic_map_cam = dataset.get_cam(rec, cam_to_world_t, cam_to_world_r)

        # plt.axis('off')  # 去坐标轴
        # plt.xticks([])  # 去 x 轴刻度
        # plt.yticks([])  # 去 y 轴刻度
        view_img_gt = view_loss(semantic_masks.unsqueeze(0).cuda(), trans.unsqueeze(0).cuda(), rots.unsqueeze(0).cuda())  # b*n 5 h w

        view_img_gt = view_img_gt.cpu().numpy()


        for j in range(6):
            plt.axis('off')  # 去坐标轴
            plt.xticks([])  # 去 x 轴刻度
            plt.yticks([])  # 去 y 轴刻度
            plt.imshow(to_image(img_origin[j]))
            #plt.show()
            plt.savefig('/data/lsy/HDmap_1/vis_gt_cam/img_{}_{}_{}'.format(idx,j,0))


            plt.axis('off')  # 去坐标轴
            plt.xticks([])  # 去 x 轴刻度
            plt.yticks([])  # 去 y 轴刻度
            plt.imshow(view_img_gt[j][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
            plt.imshow(view_img_gt[j][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
            plt.imshow(view_img_gt[j][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
            #plt.show()
            plt.savefig('/data/lsy/HDmap_1/vis_gt_cam/img_{}_{}_{}'.format(idx,j,1))

            # plt.axis('off')  # 去坐标轴
            # plt.xticks([])  # 去 x 轴刻度
            # plt.yticks([])  # 去 y 轴刻度
            # plt.imshow(semantic_map_cam[j][0][:,110:190], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
            # plt.imshow(semantic_map_cam[j][1][:,110:190], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
            # plt.imshow(semantic_map_cam[j][2][:,110:190], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
            # plt.show()

        # print('saving', idx)
        # plt.savefig('/data/lsy/HDmap_1/vis_lable/img_{}'.format(idx))
        # plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local HD Map Demo.')
    parser.add_argument('--dataroot', type=str, default='/dataset/nuScenes/v1.0')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument('--angle_class', type=int, default=36)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    args = parser.parse_args()


