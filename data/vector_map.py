import numpy as np
import torch
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
import math
from .const import CLASS2LABEL


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta):
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


def EulerAndQuaternionTransform(intput_data):
    """
        四元素与欧拉角互换
    """
    data_len = len(intput_data)
    angle_is_not_rad = False

    if data_len == 3:
        r = 0
        p = 0
        y = 0
        if angle_is_not_rad:  # 180 ->pi
            r = math.radians(intput_data[0])
            p = math.radians(intput_data[1])
            y = math.radians(intput_data[2])
        else:
            r = intput_data[0]
            p = intput_data[1]
            y = intput_data[2]

        sinp = math.sin(p / 2)
        siny = math.sin(y / 2)
        sinr = math.sin(r / 2)

        cosp = math.cos(p / 2)
        cosy = math.cos(y / 2)
        cosr = math.cos(r / 2)

        w = cosr * cosp * cosy + sinr * sinp * siny
        x = sinr * cosp * cosy - cosr * sinp * siny
        y = cosr * sinp * cosy + sinr * cosp * siny
        z = cosr * cosp * siny - sinr * sinp * cosy
        return [w, x, y, z]

    elif data_len == 4:

        w = intput_data[0]
        x = intput_data[1]
        y = intput_data[2]
        z = intput_data[3]

        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        if angle_is_not_rad:  # pi -> 180
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
        return [r, p, y]


class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num



    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
        map_pose = ego2global_translation[:2]#
        rotation = Quaternion(ego2global_rotation)
        #print('ego',map_pose)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])#[x_center, y_center, height, width]
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        #print(patch_angle)
        line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)#list 2 :
        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)['ped_crossing']

        polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, CLASS2LABEL.get(line_type, -1)))

        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, CLASS2LABEL.get('ped_crossing', -1)))

        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length, CLASS2LABEL.get('contours', -1)))

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        return filtered_vectors


    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:#每一种对象的直线数
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict#dict[road_divide:  ,lane_divied:list[5],5个对象均分点]

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)#按照顺序坐标的？
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid
