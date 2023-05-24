import argparse
import numpy as np
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import matplotlib.pyplot as plt

import tqdm
import torch

from data.dataset import semantic_dataset,my_semantic_dataset,vir_semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize
from data.viz_data import BaseViz,to_image_single,to_color,to_image
import  cv2
from loss import view_loss
def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def vis_gt(val_loader):
    viz = BaseViz()
    with torch.no_grad():
        for batchi, (img_origin, feas, imgs,  trans, rots, intrins, post_trans, post_rots,car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt) in enumerate(val_loader):

            if batchi >= 0:
                print(batchi)
                for si in range(semantic_gt.shape[0]):
                    # img = to_image(img_origin[si][1])
                    # plt.imshow(img)
                    # plt.show()
                    # img = to_image(feas[si][1])
                    # plt.imshow(img)
                    # plt.show()
                    # 展示预测结果
                    plt.figure(figsize=(4, 2), dpi=100)
                    plt.axis('off')  # 去坐标轴
                    plt.xticks([])  # 去 x 轴刻度
                    plt.yticks([])  # 去 y 轴刻度

                    plt.imshow(semantic_gt[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                    plt.imshow(semantic_gt[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                    plt.imshow(semantic_gt[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
                    #plt.imshow(semantic[si][4], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                    plt.show()

                    imname = f'eval{batchi:06}_{si:03}.jpg'
                    print('saving', imname)
                    plt.savefig('/data/lsy/HDmap_1/vis_lable_gt/img_{}_{}'.format(batchi,si))
                    plt.close()

                    # image = np.vstack(viz(semantic_gt[si], img_origin[si]))
                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # plt.axis('off')  # 去坐标轴
                    # plt.xticks([])  # 去 x 轴刻度
                    # plt.yticks([])  # 去 y 轴刻度
                    # plt.imshow(image)
                    # #plt.show()
                    # imname = f'eval{batchi:06}_{si:03}.jpg'
                    # print('saving', imname)
                    # plt.savefig('/data/lsy/HDmap_1/vis_img/img_{}_{}'.format(batchi,si))
                    # plt.close()
def vis_segmentation(model, val_loader):
    model.eval()

    with torch.no_grad():
        for batchi, (img_origin, feas, imgs,  trans, rots, intrins, post_trans, post_rots,car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt) in enumerate(val_loader):

            semantic, view_img, embedding, direction = model(feas.cuda(),imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(),car_trans.cuda(), yaw_pitch_roll.cuda())
            semantic = semantic.softmax(1).cpu().numpy()
            semantic[semantic < 0.1] = np.nan


            if batchi <51 :#and batchi < 3
                print(batchi)

                for si in range(semantic.shape[0]):


                    # 展示预测结果
                    plt.figure(figsize=(4, 2), dpi=100)
                    plt.axis('off')  # 去坐标轴
                    plt.xticks([])  # 去 x 轴刻度
                    plt.yticks([])  # 去 y 轴刻度
                    plt.imshow(semantic[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                    plt.imshow(semantic[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                    plt.imshow(semantic[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)




            else:break



def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'B': args.bsz,
        'merge': True,
    }

    #train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    val_loader = my_semantic_dataset("/data/lsy/dataset/my_dataset/ww",args.bsz, args.nworkers)
    #train_loader, val_loader = vir_semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()


    vis_segmentation(model, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='/dataset/nuScenes/v1.0')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='BiMapper')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, )

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
