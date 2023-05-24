import torch
from torch import nn

from .homography import IPM

from .base import CamEncode, BevEncode
from data.utils import gen_dx_bx
import torch.nn.functional as F

class IPMTransformation(nn.Module):
    def __init__(self ,n_views=6):
        super(IPMTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.up1 = []
        self.up2 = []


        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
            self.hw_mat.append(fc_transform)
            self.conv1 = nn.Sequential(
                nn.Conv2d(128+64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.up1.append(self.conv1)
            self.conv2 = nn.Sequential(
                nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.up2.append(self.conv2)
            # fc_transform = nn.Sequential(
            #     nn.Conv2d(768, 768, kernel_size=1, padding=0),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Conv2d(768, 768, kernel_size=1, padding=0),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            # )
            # self.hw_mat.append(fc_transform)
            # self.conv1 = nn.Sequential(
            #     nn.Conv2d(768+384, 384, kernel_size=3, padding=1, bias=False),
            #     nn.LayerNorm(384),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            #     nn.LayerNorm(384),
            #     nn.ReLU(inplace=True)
            # )
            # self.up1.append(self.conv1)
            # self.conv2 = nn.Sequential(
            #     nn.Conv2d(384 + 192, 64, kernel_size=3, padding=1, bias=False),
            #     nn.LayerNorm(64),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #     nn.LayerNorm(64),
            #     nn.ReLU(inplace=True)
            # )
            # self.up2.append(self.conv2)
        self.hw_mat = nn.ModuleList(self.hw_mat)
        self.up1 = nn.ModuleList(self.up1)
        self.up2 = nn.ModuleList(self.up2)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x1,x2,x3):
        B, N, C, H, W = x1.shape
        outputs = []
        for i in range(N):
            tt = x1[:, i] #+ self.position[i](feat[:, i])
            output_1 = self.hw_mat[i](tt)#4 128 8 22
            output_1 = self.up_sampler(output_1)#4 128 16 44
            output_2 = self.up1[i](torch.cat([output_1,x2[:,i]],dim=1))#
            output_2 = self.up_sampler(output_2)
            output_3 = self.up2[i](torch.cat([output_2,x3[:, i]], dim=1))
            outputs.append(output_3)
        outputs = torch.stack(outputs, 1)
        return outputs

class TransformModule(nn.Module):
    def __init__(self, fv_size, bv_size, n_view=6):
        super(TransformModule, self).__init__()
        self.num_view = n_view
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        self.bv_size = bv_size
        self.mat_list = nn.ModuleList()
        for i in range(self.num_view):
            fc_transform = nn.Sequential(
                        nn.Linear(fv_dim, bv_dim),
                        nn.ReLU(),
                        nn.Linear(bv_dim, bv_dim),
                        nn.ReLU()
                    )
            self.mat_list += [fc_transform]

    def forward(self, x):
        # shape x: B, V, C, H, W
        # x = x.view(list(x.size()[:3]) + [self.dim * self.dim])
        B, N, C, H, W = x.shape
        x = x.view(B, N, C, H * W)
        view_comb = self.mat_list[0](x[:, 0])
        for index in range(N)[1:]:
            view_comb = view_comb + self.mat_list[index](x[:, index])
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.bv_size[0], self.bv_size[1]])
        #view_comb = view_comb.unsqueeze(dim=1).repeat(1,self.num_view,1,1,1)
        return view_comb


class BiMapper(nn.Module):
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36,
                 lidar=False):
        super(BiMapper, self).__init__()
        self.camC = 64
        self.downsample = 16
        self.merge_flag = data_conf["merge"]
        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        self.view = 6

        self.camencode = CamEncode(self.camC)  # 初始化对象origin
        self.camencode_2 = CamEncode(self.camC)  # 初始化对象img
        fv_size = (data_conf['image_size_i'][0] // self.downsample, data_conf['image_size_i'][1] // self.downsample)  # 下采样后图片大小
        self.ipm_tran = IPMTransformation(n_views=self.view)

        bv_size_2 = (50,100)
        self.view_fore = TransformModule(fv_size=fv_size, bv_size=bv_size_2,n_view=self.view)

        ipm_xbound = [-30, 30, 0.3]  # -60 60 0.6 //200
        ipm_ybound = [-15, 15, 0.3]  # -30 30 0.6 //100

        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, extrinsic=True)

        self.up_sampler_fore = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.merge = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, padding=0),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        # 融合后上采样
        self.up_sampler_merge = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #单分支损失
        self.conv = nn.Conv2d(64, 4, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=1, padding=0)

        self.lidar = lidar

        self.i_one = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        self.f_one = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        self.mode = 'cross'
        self.loss_fn_i = torch.nn.BCEWithLogitsLoss()#内部含有sigmod
        self.loss_fn_f = torch.nn.BCEWithLogitsLoss()#内部含有sigmod

        self.bevencode_merge = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=True,
                                           embedded_dim=embedded_dim, direction_pred=direction_pred,
                                           direction_dim=direction_dim + 1)
        self.B = data_conf['B']
        self.N = 6
        self.xmin, self.xmax = -10, 10
        self.num_x = 300
        self.zmin, self.zmax = 0, 20
        self.num_z = 600


    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        # print(intrins[0][0])
        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        rots = torch.transpose(rots, dim0=2, dim1=3)
        Rs[:, :, :3, :3] = rots
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans#-trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x, x_1,x_2 = self.camencode_2(x)
        x = x.view(B, N, self.camC, imH // self.downsample, imW // self.downsample)
        x_1 = x_1.view(B, N, self.camC, 2 * imH // self.downsample, 2 * imW // self.downsample)
        x_2 = x_2.view(B, N, self.camC, 4 * imH // self.downsample, 4 * imW // self.downsample)
        return x, x_1,x_2

    def get_cam_origin_fea(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x, x_1, x_2 = self.camencode(x)
        x = x.view(B, N, self.camC, imH // self.downsample, imW // self.downsample)
        x_1 = x_1.view(B, N, self.camC, 2 * imH // self.downsample, 2 * imW // self.downsample)
        x_2 = x_2.view(B, N, self.camC, 4 * imH // self.downsample, 4 * imW // self.downsample)
        return x, x_1, x_2


    def cam_loss(self,v):

        v2 = self.conv(v)
        v3 = self.conv1(v2)
        return v2,v3+v


    def mutual_loss(self,v_i,v_f):
        if self.mode=="kl" :
            i_m = torch.sum(v_i, dim=1)  # 4 100 200
            f_m = torch.sum(v_f, dim=1)

            B, h, w = i_m.shape
            i_m_flat = i_m.view(B, h * w)
            i_m_soft = F.softmax(i_m_flat, dim=-1)
            i_m_soft = i_m_soft.view(B, h, w)

            f_m_flat = f_m.view(B, h * w)
            f_m_soft = F.softmax(f_m_flat, dim=-1)
            f_m_soft = f_m_soft.view(B, h, w)

            mask = torch.ones_like(i_m_soft) * 1e-7
            i_m_soft = i_m_soft + mask
            f_m_soft = f_m_soft + mask
            kl_2 = F.kl_div(f_m_soft.log(), i_m_soft, reduction='mean')  # 第一个为预测分布，第二个为真实分布
            kl_1 = F.kl_div(i_m_soft.log(), f_m_soft, reduction='mean')  # 第一个为预测分布，第二个为真实分布
        if self.mode == "cross":
            i_m = self.i_one(v_i)
            f_m = self.f_one(v_f)#4 2 100 200
            i_m_soft = F.softmax(i_m,dim=1)
            f_m_soft = F.softmax(f_m, dim=1)
            i_m_mask = torch.ones_like(i_m_soft)
            i_m_mask[:,0] = i_m_soft[:,0]>0.5
            i_m_mask[:,1] = i_m_soft[:, 1] > 0.5
            kl_2 = self.loss_fn_i(f_m_soft,i_m_mask)
            f_m_mask = torch.ones_like(f_m_soft)
            f_m_mask[:, 0] = f_m_soft[:, 0] > 0.5
            f_m_mask[:, 1] = f_m_soft[:, 1] > 0.5
            kl_1 = self.loss_fn_f(i_m_soft, f_m_mask)#vpn为真值

        return kl_1,kl_2

    def forward(self, img_origin, img, trans, rots, intrins, post_trans, post_rots, car_trans,
                yaw_pitch_roll):


        x_31, x_41,x_51 = self.get_cam_origin_fea(img_origin)  # 4 6 64 8 22plane_fea_1.contiguous()
        x_5 = self.ipm_tran(x_31, x_41,x_51)  # 4 6 64 32 88

        x, x_1,x_2 = self.get_cam_feats(img)  # 4 6 64 8 22
        x_6 = self.view_fore(x)# 4 64 50 100
        x_6 = self.up_sampler_fore(x_6)# 100 200


        b,n,c,h,w = x_5.shape
        x_5 = x_5.view(-1,c,h,w)
        x_view_l,x_51 = self.cam_loss(x_5)
        x_51 = x_51.view(b, n, c, h, w)


        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        x_71 = self.ipm(x_51, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)#100,200


        if True:
            topdown = 0.1*x_71 + x_6  #
            topdown = self.merge(topdown)#xg
            mu_loss,mu_loss_i = self.mutual_loss(x_71,x_6)
            topdown = self.up_sampler_merge(topdown)  # 200,400
            semantic,x1, direction = self.bevencode_merge(topdown)#



        return semantic,x_view_l,mu_loss,mu_loss_i#,mu_loss_m#,embedding#, direction






