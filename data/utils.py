import numpy as np
import xml.etree.ElementTree as xmlET
import torch
import os
import math

def get_proj_mat(intrins, rots, trans):
    K = np.eye(4)
    K[:3, :3] = intrins
    R = np.eye(4)
    R[:3, :3] = rots.transpose(-1, -2)
    T = np.eye(4)
    T[:3, 3] = -trans
    RT = R @ T
    return K @ RT


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def label_onehot_decoding(onehot):
    return torch.argmax(onehot, axis=0)


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def eulerAngles2rotationMat(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
def get_files_in_folder(folder):

    return sorted([os.path.join(folder, f) for f in os.listdir(folder)])#返回图片名字路径

def one_hot_encode_image(image, palette):

    one_hot_map = []

    # find instances of class colors and append layer to one-hot-map
    for class_colors in palette:
        class_map = np.zeros(image.shape[0:2], dtype=bool)
        for color in class_colors:
            # temp = np.ones_like(image)
            # temp  = temp [:,:,0] * color[0]
            # temp = temp[:, :, 1] * color[1]
            # temp = temp[:, :, 2] * color[2]
            class_map = class_map | (image==color).all(axis=-1)
        one_hot_map.append(class_map)

    # finalize one-hot-map
    one_hot_map = np.stack(one_hot_map, axis=-1)
    one_hot_map = one_hot_map.astype(np.float32)

    return one_hot_map
def parse_convert_xml(conversion_file_path):

    defRoot = xmlET.parse(conversion_file_path).getroot()

    one_hot_palette = []
    class_list = []
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        from_color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        to_class = np.fromstring(defElement.get("toValue"), dtype=int, sep=" ")
        if to_class in class_list:
             one_hot_palette[class_list.index(to_class)].append(from_color)
        else:
            one_hot_palette.append([from_color])
            class_list.append(to_class)

    return one_hot_palette
