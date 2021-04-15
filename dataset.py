import torch
import torch_geometric.transforms as transforms
from torch_geometric.data import Data, InMemoryDataset

import os
import math
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
from utils.urdf2graph import yumi2graph
import h5py


"""
Normalize by a constant coefficient
"""
class Normalize(object):
    def __call__(self, data, coeff=100.0):
        data.x = data.x/coeff
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

"""
Target Dataset for Yumi Manipulator
"""
class YumiDataset(InMemoryDataset):
    yumi_cfg = {
        'joints_name': [
            'yumi_joint_1_l',
            'yumi_joint_2_l',
            'yumi_joint_7_l',
            'yumi_joint_3_l',
            'yumi_joint_4_l',
            'yumi_joint_5_l',
            'yumi_joint_6_l',
            'yumi_joint_1_r',
            'yumi_joint_2_r',
            'yumi_joint_7_r',
            'yumi_joint_3_r',
            'yumi_joint_4_r',
            'yumi_joint_5_r',
            'yumi_joint_6_r',
        ],
        'edges': [
            ['yumi_joint_1_l', 'yumi_joint_2_l'],
            ['yumi_joint_2_l', 'yumi_joint_7_l'],
            ['yumi_joint_7_l', 'yumi_joint_3_l'],
            ['yumi_joint_3_l', 'yumi_joint_4_l'],
            ['yumi_joint_4_l', 'yumi_joint_5_l'],
            ['yumi_joint_5_l', 'yumi_joint_6_l'],
            ['yumi_joint_1_r', 'yumi_joint_2_r'],
            ['yumi_joint_2_r', 'yumi_joint_7_r'],
            ['yumi_joint_7_r', 'yumi_joint_3_r'],
            ['yumi_joint_3_r', 'yumi_joint_4_r'],
            ['yumi_joint_4_r', 'yumi_joint_5_r'],
            ['yumi_joint_5_r', 'yumi_joint_6_r'],
        ],
        'root_name': [
            'yumi_joint_1_l',
            'yumi_joint_1_r',
        ],
        'end_effectors': [
            'yumi_joint_6_l',
            'yumi_joint_6_r',
        ],
        'shoulders': [
            'yumi_joint_2_l',
            'yumi_joint_2_r',
        ],
        'elbows': [
            'yumi_joint_3_l',
            'yumi_joint_3_r',
        ],
    }
    def __init__(self, root, transform=None, pre_transform=None):
        super(YumiDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        self._raw_file_names = [os.path.join(self.root, file) for file in os.listdir(self.root) if file.endswith('.urdf')]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for file in self.raw_file_names:
            data_list.append(yumi2graph(file, self.yumi_cfg))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
Map glove data to inspire hand data
"""
def linear_map(x_, min_, max_, min_hat, max_hat):
    x_hat = 1.0 * (x_ - min_) / (max_ - min_) * (max_hat - min_hat) + min_hat
    return x_hat

def map_glove_to_inspire_hand(glove_angles):

    ### This function linearly maps the Wiseglove angle measurement to Inspire hand's joint angles.

    ## preparation, specify the range for linear scaling
    hand_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0, 0.0]) # radius already
    hand_final = np.array([-1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -0.75, 0.0, -0.2, -0.15])
    glove_start = np.array([0, 0, 53, 0, 0, 22, 0, 0, 22, 0, 0, 35, 0, 0])# * pi / 180.0 # degree to radius
    glove_final = np.array([45, 100, 0, 90, 120, 0, 90, 120, 0, 90, 120, 0, 90, 120])# * pi / 180.0
    length = glove_angles.shape[0]
    hand_angles = np.zeros((length, 12)) # 12 joints

    ## Iterate to map angles
    for i in range(length):
        # four fingers' extension/flexion (abduction/adduction are dumped)
        hand_angles[i, 0] = linear_map(glove_angles[i, 3], glove_start[3], glove_final[3], hand_start[0], hand_final[0]) # Link1 (joint name)
        hand_angles[i, 1] = linear_map(glove_angles[i, 4], glove_start[4], glove_final[4], hand_start[1], hand_final[1]) # Link11
        hand_angles[i, 2] = linear_map(glove_angles[i, 6], glove_start[6], glove_final[6], hand_start[2], hand_final[2]) # Link2
        hand_angles[i, 3] = linear_map(glove_angles[i, 7], glove_start[7], glove_final[7], hand_start[3], hand_final[3]) # Link22
        hand_angles[i, 4] = linear_map(glove_angles[i, 9], glove_start[9], glove_final[9], hand_start[4], hand_final[4]) # Link3
        hand_angles[i, 5] = linear_map(glove_angles[i, 10], glove_start[10], glove_final[10], hand_start[5], hand_final[5]) # Link33
        hand_angles[i, 6] = linear_map(glove_angles[i, 12], glove_start[12], glove_final[12], hand_start[6], hand_final[6]) # Link4
        hand_angles[i, 7] = linear_map(glove_angles[i, 13], glove_start[13], glove_final[13], hand_start[7], hand_final[7]) # Link44

        # thumb
        hand_angles[i, 8] = (hand_start[8] + hand_final[8]) / 2.0 # Link5 (rotation about z axis), fixed!
        hand_angles[i, 9] = linear_map(glove_angles[i, 2], glove_start[2], glove_final[2], hand_start[9], hand_final[9]) # Link 51
        hand_angles[i, 10] = linear_map(glove_angles[i, 0], glove_start[0], glove_final[0], hand_start[10], hand_final[10]) # Link 52
        hand_angles[i, 11] = linear_map(glove_angles[i, 1], glove_start[1], glove_final[1], hand_start[11], hand_final[11]) # Link 53

    return hand_angles

"""
Parse H5 File
"""
def parse_h5(filename, selected_key=None):
    data_list = []
    h5_file = h5py.File(filename, 'r')
    # print(filename, h5_file.keys(), len(h5_file.keys()))
    if selected_key is None:
        keys = h5_file.keys()
    else:
        keys = [selected_key]
    for key in keys:
        if '语句' in key and selected_key is None:
            print('Skipping'+key)
            continue
        # glove data
        l_glove_angle = h5_file[key + '/l_glove_angle'][:]
        r_glove_angle = h5_file[key + '/r_glove_angle'][:]
        l_hand_angle = map_glove_to_inspire_hand(l_glove_angle)
        r_hand_angle = map_glove_to_inspire_hand(r_glove_angle)
        # position data
        l_shoulder_pos = h5_file[key + '/l_up_pos'][:]
        r_shoulder_pos = h5_file[key + '/r_up_pos'][:]
        l_elbow_pos = h5_file[key + '/l_fr_pos'][:]
        r_elbow_pos = h5_file[key + '/r_fr_pos'][:]
        l_wrist_pos = h5_file[key + '/l_hd_pos'][:]
        r_wrist_pos = h5_file[key + '/r_hd_pos'][:]
        # quaternion data
        l_shoulder_quat = R.from_quat(h5_file[key + '/l_up_quat'][:])
        r_shoulder_quat = R.from_quat(h5_file[key + '/r_up_quat'][:])
        l_elbow_quat = R.from_quat(h5_file[key + '/l_fr_quat'][:])
        r_elbow_quat = R.from_quat(h5_file[key + '/r_fr_quat'][:])
        l_wrist_quat = R.from_quat(h5_file[key + '/l_hd_quat'][:])
        r_wrist_quat = R.from_quat(h5_file[key + '/r_hd_quat'][:])
        # rotation matrix data
        l_shoulder_matrix = l_shoulder_quat.as_matrix()
        r_shoulder_matrix = r_shoulder_quat.as_matrix()
        l_elbow_matrix = l_elbow_quat.as_matrix()
        r_elbow_matrix = r_elbow_quat.as_matrix()
        l_wrist_matrix = l_wrist_quat.as_matrix()
        r_wrist_matrix = r_wrist_quat.as_matrix()
        # transform to local coordinates
        l_wrist_matrix = l_wrist_matrix * inv(l_elbow_matrix)
        r_wrist_matrix = r_wrist_matrix * inv(r_elbow_matrix)
        l_elbow_matrix = l_elbow_matrix * inv(l_shoulder_matrix)
        r_elbow_matrix = r_elbow_matrix * inv(r_shoulder_matrix)
        # l_shoulder_matrix = l_shoulder_matrix * inv(l_shoulder_matrix)
        # r_shoulder_matrix = r_shoulder_matrix * inv(r_shoulder_matrix)
        # euler data
        l_shoulder_euler = R.from_matrix(l_wrist_matrix).as_euler('zyx', degrees=True)
        r_shoulder_euler = R.from_matrix(r_wrist_matrix).as_euler('zyx', degrees=True)
        l_elbow_euler = R.from_matrix(l_elbow_matrix).as_euler('zyx', degrees=True)
        r_elbow_euler = R.from_matrix(r_elbow_matrix).as_euler('zyx', degrees=True)
        l_wrist_euler = R.from_matrix(l_shoulder_matrix).as_euler('zyx', degrees=True)
        r_wrist_euler = R.from_matrix(r_shoulder_matrix).as_euler('zyx', degrees=True)

        total_frames = l_shoulder_pos.shape[0]
        for t in range(total_frames):
            # x
            x = torch.stack([torch.from_numpy(l_shoulder_euler[t]),
                             torch.from_numpy(l_elbow_euler[t]),
                             torch.from_numpy(l_wrist_euler[t]),
                             torch.from_numpy(r_shoulder_euler[t]),
                             torch.from_numpy(r_elbow_euler[t]),
                             torch.from_numpy(r_wrist_euler[t])], dim=0).float()
            # number of nodes
            num_nodes = 6
            # edge index
            edge_index = torch.LongTensor([[0, 1, 3, 4],
                                           [1, 2, 4, 5]])
            # position
            pos = torch.stack([torch.from_numpy(l_shoulder_pos[t]),
                               torch.from_numpy(l_elbow_pos[t]),
                               torch.from_numpy(l_wrist_pos[t]),
                               torch.from_numpy(r_shoulder_pos[t]),
                               torch.from_numpy(r_elbow_pos[t]),
                               torch.from_numpy(r_wrist_pos[t])], dim=0).float()
            # edge attributes
            edge_attr = []
            for edge in edge_index.permute(1, 0):
                parent = edge[0]
                child = edge[1]
                edge_attr.append(pos[child] - pos[parent])
            edge_attr = torch.stack(edge_attr, dim=0)
            # skeleton type & topology type
            skeleton_type = 0
            topology_type = 0
            # end effector mask
            ee_mask = torch.zeros(num_nodes, 1).bool()
            ee_mask[2] = ee_mask[5] = True
            # shoulder mask
            sh_mask = torch.zeros(num_nodes, 1).bool()
            sh_mask[0] = sh_mask[3] = True
            # elbow mask
            el_mask = torch.zeros(num_nodes, 1).bool()
            el_mask[1] = el_mask[4] = True
            # parent
            parent = torch.LongTensor([-1, 0, 1, -1, 3, 4])
            # offset
            offset = torch.zeros(num_nodes, 3)
            for node_idx in range(num_nodes):
                if parent[node_idx] != -1:
                    offset[node_idx] = pos[node_idx] - pos[parent[node_idx]]
                else:
                    offset[node_idx] = pos[node_idx]
            # distance to root
            root_dist = torch.zeros(num_nodes, 1)
            for node_idx in range(num_nodes):
                dist = 0
                current_idx = node_idx
                while current_idx != -1:
                    origin = offset[current_idx]
                    offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
                    dist += offsets_mod
                    current_idx = parent[current_idx]
                root_dist[node_idx] = dist
            # distance to shoulder
            shoulder_dist = torch.zeros(num_nodes, 1)
            for node_idx in range(num_nodes):
                dist = 0
                current_idx = node_idx
                while current_idx != -1 and current_idx != 0 and current_idx != 3:
                    origin = offset[current_idx]
                    offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
                    dist += offsets_mod
                    current_idx = parent[current_idx]
                shoulder_dist[node_idx] = dist
            # distance to elbow
            elbow_dist = torch.zeros(num_nodes, 1)
            for node_idx in range(num_nodes):
                dist = 0
                current_idx = node_idx
                while current_idx != -1 and current_idx != 1 and current_idx != 4:
                    origin = offset[current_idx]
                    offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
                    dist += offsets_mod
                    current_idx = parent[current_idx]
                elbow_dist[node_idx] = dist
            # quaternion
            q = torch.stack([torch.from_numpy(h5_file[key + '/l_up_quat'][t]),
                             torch.from_numpy(h5_file[key + '/l_fr_quat'][t]),
                             torch.from_numpy(h5_file[key + '/l_hd_quat'][t]),
                             torch.from_numpy(h5_file[key + '/r_up_quat'][t]),
                             torch.from_numpy(h5_file[key + '/r_fr_quat'][t]),
                             torch.from_numpy(h5_file[key + '/r_hd_quat'][t])], dim=0).float()
            data = Data(x=torch.cat([x,pos], dim=-1),
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=pos,
                        q=q,
                        skeleton_type=skeleton_type,
                        topology_type=topology_type,
                        ee_mask=ee_mask,
                        sh_mask=sh_mask,
                        el_mask=el_mask,
                        root_dist=root_dist,
                        shoulder_dist=shoulder_dist,
                        elbow_dist=elbow_dist,
                        num_nodes=num_nodes,
                        parent=parent,
                        offset=offset)
            # print(data)
            data_list.append(data)
    return data_list, l_hand_angle, r_hand_angle

"""
Source Dataset for Sign Language
"""
class SignDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SignDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_path = os.path.join(self.root, 'h5')
        self._raw_file_names = [os.path.join(data_path, file) for file in os.listdir(data_path)]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for file in self.raw_file_names:
            data, _, _ = parse_h5(file)
            data_list.extend(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    yumi_dataset = YumiDataset(root='./data/target/yumi')
    sign_dataset = SignDataset(root='./data/source/yumi/train', pre_transform=transforms.Compose([Normalize()]))
