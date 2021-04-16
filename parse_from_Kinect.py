import torch
from torch_geometric.data import Data
import math
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import h5py
from body_movement_vis import pyKinect
import numpy as np
import body_movement_vis.visualize_sign as visualize_sign
import threading
# from h5_control_single import kinect_pos as kpos
# from h5_control_single import kinect_quat as kquat
# import h5_control_single
# from h5_control_single import return_pos_quat
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

class vis(threading.Thread):
    def __init__(self,threadID,name,counter,lib):
        threading.Thread.__init__(self)
        self.threadID=threadID
        self.name=name
        self.counte=counter
        self.lib=lib
    def run(self):
        visualize_sign.parse_h5(self.lib)

"""
Parse H5 File
"""
def parse_from_Kinect(pos,quat,filename, selected_key=None):
    data_list = []
    h5_file = h5py.File(filename, 'r')
    # print(filename, h5_file.keys(), len(h5_file.keys()))
    if selected_key is None:
        keys = h5_file.keys()
    else:
        keys = [selected_key]
    for key in keys:
        if '语句' in key:
            print('Skipping'+key)
            continue
        # glove data
        l_glove_angle = h5_file[key + '/l_glove_angle'][1:2]#shape(Tx15)
        r_glove_angle = h5_file[key + '/r_glove_angle'][1:2]#shape(Tx15)
        l_hand_angle = map_glove_to_inspire_hand(l_glove_angle)
        r_hand_angle = map_glove_to_inspire_hand(r_glove_angle)

        #read data from kinect
        # kinect_pos,kinect_quat=pyKinect.get_data()
        kinect_pos, kinect_quat = pos,quat

        # position data         shape(Tx3)
        l_shoulder_pos = torch.tensor(kinect_pos[0])
        r_shoulder_pos = torch.tensor(kinect_pos[1])

        origin = (l_shoulder_pos+r_shoulder_pos)/2 #calculate origin point
        # define rotation matrix
        rot_mat = torch.tensor([[0, 1, 0],
                                [0, 0, -1],
                                [-1, 0, 0]]).float()

        l_shoulder_pos = ((l_shoulder_pos-origin).matmul(rot_mat))/1000
        r_shoulder_pos = ((r_shoulder_pos-origin).matmul(rot_mat))/1000
        l_elbow_pos = ((torch.tensor(kinect_pos[2])-origin).matmul(rot_mat))/1000
        r_elbow_pos = ((torch.tensor(kinect_pos[3])-origin).matmul(rot_mat))/1000
        l_wrist_pos = ((torch.tensor(kinect_pos[4])-origin).matmul(rot_mat))/1000
        r_wrist_pos = ((torch.tensor(kinect_pos[5])-origin).matmul(rot_mat))/1000

        # #vis
        # thread_vis=vis(1,"thread_vis",1,lib)
        # thread_vis.start()
        # thread_vis.join()

        # quaternion data       shape(Tx4)    动捕录下来的四元数是在世界坐标系下的吗？
        l_shoulder_quat = R.from_quat(kinect_quat[0])
        r_shoulder_quat = R.from_quat(kinect_quat[1])
        l_elbow_quat = R.from_quat(kinect_quat[2])
        r_elbow_quat = R.from_quat(kinect_quat[3])
        l_wrist_quat = R.from_quat(kinect_quat[4])
        r_wrist_quat = R.from_quat(kinect_quat[5])

        # euler data
        # l_shoulder_euler = l_shoulder_quat.as_euler('xyz', degrees=True)
        # r_shoulder_euler = r_shoulder_quat.as_euler('xyz', degrees=True)
        # l_elbow_euler = l_elbow_quat.as_euler('xyz', degrees=True)
        # r_elbow_euler = r_elbow_quat.as_euler('xyz', degrees=True)
        # l_wrist_euler = l_wrist_quat.as_euler('xyz', degrees=True)
        # r_wrist_euler = r_wrist_quat.as_euler('xyz', degrees=True)
        # print(l_shoulder_pos.shape, r_shoulder_pos.shape, l_elbow_pos.shape, r_elbow_pos.shape, l_wrist_pos.shape, r_wrist_pos.shape)

        # rotation matrix data    四元数转换为3x3旋转矩阵？
        l_shoulder_matrix = l_shoulder_quat.as_matrix()
        r_shoulder_matrix = r_shoulder_quat.as_matrix()
        l_elbow_matrix = l_elbow_quat.as_matrix()
        r_elbow_matrix = r_elbow_quat.as_matrix()
        l_wrist_matrix = l_wrist_quat.as_matrix()
        r_wrist_matrix = r_wrist_quat.as_matrix()

        # transform to local coordinates    ？
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

        total_frames = 1
        for t in range(total_frames):
            # x
            x = torch.stack([torch.from_numpy(l_shoulder_euler),
                             torch.from_numpy(l_elbow_euler),
                             torch.from_numpy(l_wrist_euler),
                             torch.from_numpy(r_shoulder_euler),
                             torch.from_numpy(r_elbow_euler),
                             torch.from_numpy(r_wrist_euler)], dim=0).float()
            # number of nodes
            num_nodes = 6
            # edge index
            edge_index = torch.LongTensor([[0, 1, 3, 4],
                                           [1, 2, 4, 5]])
            # edge_index = torch.LongTensor([[0, 2, 1, 3],
            #                                [2, 4, 3, 5]])
            # position
            pos = torch.stack([torch.from_numpy(np.array(l_shoulder_pos)),
                               torch.from_numpy(np.array(l_elbow_pos)),
                               torch.from_numpy(np.array(l_wrist_pos)),
                               torch.from_numpy(np.array(r_shoulder_pos)),
                               torch.from_numpy(np.array(r_elbow_pos)),
                               torch.from_numpy(np.array(r_wrist_pos))], dim=0).float()
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
            #parse_h5这个函数在SignDataset类里用到了，作用是从h5格式的source motion中解析出数据，然后按照下面的方式来组织成图。
            data = Data(x=torch.cat([x,pos], dim=-1),#节点特征
                        edge_index=edge_index,#边连接关系
                        edge_attr=edge_attr,#边特征
                        pos=pos,#点的位置，为了之后算loss使用
                        q=q,#？
                        skeleton_type=skeleton_type,
                        topology_type=topology_type,
                        ee_mask=ee_mask,#末端mask（通过7x1的tensor来选出是末端的节点）
                        sh_mask=sh_mask,#shoulder mask
                        el_mask=el_mask,#elbow mask
                        root_dist=root_dist,#每个点到root的距离（之后算loss会用到）
                        shoulder_dist=shoulder_dist,#同理
                        elbow_dist=elbow_dist,#同理
                        num_nodes=num_nodes,
                        parent=parent,#每个节点的前驱
                        offset=offset)#每个节点相对前驱的偏移（具体是什么？）
            # print(data)
            data_list.append(data)
        break
    return data_list, l_hand_angle, r_hand_angle
