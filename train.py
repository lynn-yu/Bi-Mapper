import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse
import cv2
import torch
from torch.optim.lr_scheduler import StepLR
from loss import SimpleLoss, DiscriminativeLoss, view_loss,FocalLoss
import torch.nn as nn
from data.dataset import semantic_dataset,vir_semantic_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from evaluate import onehot_encoding, eval_iou

def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)
def train(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'image_size_i': args.image_size_i,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
        'B':args.bsz,
        'merge': True
    }
    #train_loader, val_loader = vir_semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)#构建训练和验证集
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)#选择不同传感器布局方式的网络
    read_last = args.read_last
    if read_last:#args.finetune
        model.load_state_dict(torch.load(args.modelf), strict=False)
        for name, param in model.named_parameters():
            param.requires_grad = True


    model.cuda()

    opt = torch.optim.AdamW([{'params': model.parameters(), 'initial_lr': 1e-3}], lr=1e-3, weight_decay=args.weight_decay)


    if read_last:
        sched = StepLR(opt, 10, 0.1, last_epoch=read_last-1)
    else:
        sched = StepLR(opt, 10, 0.1)

    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).cuda()#损失对象
    loss_fn_view = SimpleLoss(args.pos_weight).cuda()  # 损失对象



    model.train()
    counter = 0
    last_idx = len(train_loader) - 1


    for epoch in range(read_last,args.nepochs):
        sched.last_epoch = -1
        for batchi, (img_origin, feas, imgs,  trans, rots, intrins, post_trans, post_rots,car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt) in enumerate(train_loader):#, direction_gt
            t0 = time()


            opt.zero_grad()
            semantic , x_view_pre, mu_loss, mu_loss_i= model(feas.cuda(),imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                   post_trans.cuda(), post_rots.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())


            semantic_gt = semantic_gt.cuda().float()


            view_img_gt = view_loss(semantic_gt.cuda(),trans.cuda(),rots.cuda())#b*n 5 h w
            view_img_loss =  loss_fn_view(x_view_pre,view_img_gt)
            seg_loss = loss_fn(semantic, semantic_gt)


            if epoch<5:
                final_loss = seg_loss * args.scale_seg + view_img_loss +0.1*mu_loss_i#+seg_loss_f#+ var_loss * args.scale_var + dist_loss * args.scale_dist+ mu_loss# + direction_loss * args.scale_direction

            else:
                final_loss = seg_loss * args.scale_seg + view_img_loss + 0.1*mu_loss_i+ 0.1*mu_loss

            final_loss.requires_grad_(True)
            final_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            opt.step()
            counter += 1
            t1 = time()
            sched.step()

            if counter % 10 == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)
                intersects1, union1 = get_batch_iou(onehot_encoding(x_view_pre), view_img_gt)
                iou_cam = intersects1 / (union1 + 1e-7)

                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1-t0:>7.4f}    "
                            f"Loss: [{final_loss.item():>7.4f}],[{seg_loss.item():>7.4f}],[{view_img_loss.item():>7.4f}]  "#
                            f"mutual_Loss:[{mu_loss_i:>7.4f}],[{mu_loss.item():>7.4f}]"
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}   "
                            f"IOU_cam: {np.array2string(iou_cam[1:].numpy(), precision=3, floatmode='fixed')}  "
                            )

                write_log(writer, iou, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)


        iou,iou_cam = eval_iou(model, val_loader)
        logger.info(f"EVAL[{epoch:>2d}]:    "
                        f"IOU: {np.array2string(iou[1:].cpu().numpy(), precision=3, floatmode='fixed')}  "
                        f"IOU_cam: {np.array2string(iou_cam[1:].cpu().numpy(), precision=3, floatmode='fixed')}"
                        )
        write_log(writer, iou, 'eval', counter)

        model_name = os.path.join(args.logdir, f"model{epoch}.pt")
        torch.save(model.state_dict(), model_name)
        logger.info(f"{model_name} saved")
        model.train()

        #sched.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')#

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='/dataset/nuScenes/v1.0')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--read_last', type=int, default=0)
    # model config
    parser.add_argument("--model", type=str, default='BiMapper')

    # training config
    parser.add_argument("--nepochs", type=int, default=24)
    parser.add_argument("--max_grad_norm", type=float, default=35.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 704])
    parser.add_argument("--image_size_i", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])#200
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
    train(args)
