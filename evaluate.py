import argparse
import tqdm
import os
import torch
from data.dataset import semantic_dataset,mydataset,vir_semantic_dataset
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from model import get_model

from loss import view_loss

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def eval_iou(model, val_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    total_intersects1 = 0
    total_union1 = 0
    with torch.no_grad():
        for img_origin, feas, imgs, trans, rots, intrins, post_trans, post_rots,  car_trans, yaw_pitch_roll, semantic_gt, instance_gt in tqdm.tqdm(val_loader):

            semantic, x_view_pre,view_ipm, viw_fore = model(feas.cuda(),imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
            #, embedding, direction
            semantic_gt = semantic_gt.cuda().float()
            intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
            view_img_gt = view_loss(semantic_gt.cuda(), trans.cuda(), rots.cuda())  # b*n 5 h w
            intersects1, union1 = get_batch_iou(onehot_encoding(x_view_pre), view_img_gt)
            total_intersects1 += intersects1
            total_union1 += union1
    return total_intersects / (total_union + 1e-7),total_intersects1 / (total_union1 + 1e-7)

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
        'merge': True
    }

    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    #train_loader, val_loader = vir_semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)

    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    res,_ = eval_iou(model,val_loader)
    print(res)



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
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', default=False, action='store_true')
    parser.add_argument('--modelf', type=str, default='')

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
