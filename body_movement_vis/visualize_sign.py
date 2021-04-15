import h5py
import math
from scipy.spatial.transform import Rotation as R
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from body_movement_vis import pyKinect
# -*- coding: utf-8 -*-
from ctypes import *
import time

# 指定动态链接库
# lib = cdll.LoadLibrary('./read.so')
# lib.get_data.restype = POINTER(c_float)

def parse_h5(pos):
    # data_list = []
    # h5_file = h5py.File(filename, 'r')
    # print(filename, h5_file.keys())
    def get_data():
        # pos,quat=pyKinect.get_data()      #if want to show vis of raw data, uncomment this
        l_shoulder_pos = pos[0]
        r_shoulder_pos = pos[1]
        l_elbow_pos = pos[2]
        r_elbow_pos = pos[3]
        l_wrist_pos = pos[4]
        r_wrist_pos = pos[5]


        l_shoulder_pos = torch.tensor(l_shoulder_pos)
        r_shoulder_pos = torch.tensor(r_shoulder_pos)
        origin = (l_shoulder_pos + r_shoulder_pos) / 2
        rot_mat = torch.tensor([[0, 1, 0],
                                [0, 0, -1],
                                [-1, 0, 0]]).float()
        l_shoulder_pos = ((l_shoulder_pos - origin).matmul(rot_mat)           )/1000
        r_shoulder_pos = ((r_shoulder_pos - origin).matmul(rot_mat)           )/1000
        l_elbow_pos =    ((torch.tensor(l_elbow_pos) - origin).matmul(rot_mat))/1000
        r_elbow_pos =    ((torch.tensor(r_elbow_pos) - origin).matmul(rot_mat))/1000
        l_wrist_pos =    ((torch.tensor(l_wrist_pos) - origin).matmul(rot_mat))/1000
        r_wrist_pos =    ((torch.tensor(r_wrist_pos) - origin).matmul(rot_mat))/1000
        return l_shoulder_pos,r_shoulder_pos,l_elbow_pos,r_elbow_pos,l_wrist_pos,r_wrist_pos

    def run(t):

        a = []
        l_shoulder_pos, r_shoulder_pos, l_elbow_pos, r_elbow_pos, l_wrist_pos, r_wrist_pos=get_data()
        a.append(l_shoulder_pos.tolist())
        a.append(r_shoulder_pos.tolist())
        a.append(l_elbow_pos   .tolist())
        a.append(r_elbow_pos   .tolist())
        a.append(l_wrist_pos   .tolist())
        a.append(r_wrist_pos   .tolist())

        pos = torch.tensor(a)
        edge_index = torch.tensor([[0, 2, 1, 3], [2, 4, 3, 5]])
        # edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])

        for line, edge in zip(lines, edge_index.permute(1, 0)):
            line_x = [pos[edge[0]][0], pos[edge[1]][0]]
            line_y = [pos[edge[0]][1], pos[edge[1]][1]]
            line_z = [pos[edge[0]][2], pos[edge[1]][2]]
            line.set_data(np.array([line_x, line_y]))
            line.set_3d_properties(np.array(line_z))
        return lines

    # attach 3D axis to figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(elev=0, azim=0)
    # set axis limits & labels
    ax.set_xlim3d([-0.2, 0.2])
    ax.set_xlabel('X')
    ax.set_ylim3d([-0.4, 0.4])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-0.5, 0])
    ax.set_zlabel('Z')
    # create animation
    lines = [ax.plot([], [], [], 'royalblue', marker='o')[0] for i in range(4)]
    ani = animation.FuncAnimation(fig, run, interval=50,cache_frame_data=False)
    plt.show()

if __name__ == '__main__':
    pyKinect.init()
    parse_h5(1)
    pyKinect.deinit()