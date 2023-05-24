import numpy as np
import cv2

from pathlib import Path
from pyquaternion import Quaternion


INTERPOLATION = cv2.LINE_8


def get_split(split, dataset_name):
    split_dir = Path(__file__).parent / 'splits' / dataset_name
    split_path = split_dir / f'{split}.txt'

    return split_path.read_text().strip().split('\n')#返回场景序号列表[scenes-0001,....]


def get_view_matrix(h=200, w=400, h_meters=30.0, w_meters=60.0, h_offset=2, w_offset=-2):
    sh = h / h_meters
    sw = w / w_meters
    t1 = np.float32([
        [ 0., sw, w*w_offset+w/2.],
        [sh,  0., h*h_offset+h/2.],
        [ 0.,  0.,            1.]
    ])
    t2 = np.float32([
        [ sw, 0.,  w*w_offset+w/2.],
        [0.,  sh,  h*h_offset+h/2.],
        [ 0.,  0.,            1.]
    ])
    return t2


def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose


def get_pose(rotation, translation, inv=False, flat=False):
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)

    return get_transformation_matrix(R, t, inv=inv)


def encode(x):
    """
    (h, w, c) np.uint8 {0, 255}
    """
    n = x.shape[2]

    # assert n < 16
    assert x.ndim == 3
    assert x.dtype == np.uint8
    assert all(x in [0, 255] for x in np.unique(x))

    shift = np.arange(n, dtype=np.int32)[None, None]

    binary = (x > 0)
    binary = (binary << shift).sum(-1)
    binary = binary.astype(np.int32)

    return binary


def decode(img, n):
    """
    returns (h, w, n) np.int32 {0, 1}
    """
    shift = np.arange(n, dtype=np.int32)[None, None]

    x = np.array(img)[..., None]
    x = (x >> shift) & 1

    return x


if __name__ == '__main__':
    from PIL import Image

    n = 12

    x = np.random.rand(64, 64, n)
    x = 255 * (x > 0.5).astype(np.uint8)

    x_encoded = encode(x)
    x_img = Image.fromarray(x_encoded)
    x_img.save('tmp.png')
    x_loaded = Image.open('tmp.png')
    x_decoded = 255 * decode(x_loaded, 12)
    x_decoded = x_decoded[..., :n]

    print(abs(x_decoded - x).max())
