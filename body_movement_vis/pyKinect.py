# -*- coding: utf-8 -*-
from ctypes import *
import time
import torch
import ctypes
class StructPointer(Structure):
    _fields_ = [("pos", c_float * 3),
                ("quat", c_float * 4)]

# 指定动态链接库
lib = cdll.LoadLibrary('/home/yu/PycharmProjects/MotionTransfer-master-Yu-comment/body_movement_vis/read.so')
lib.get_data.restype = POINTER(POINTER(c_float))
# lib.get_data.restype = c_float * 2
'''      joint ID index (https://docs.microsoft.com/en-us/azure/kinect-dk/body-joints)
5	SHOULDER_LEFT	CLAVICLE_LEFT
6	ELBOW_LEFT	SHOULDER_LEFT
7	WRIST_LEFT	ELBOW_LEFT

12	SHOULDER_RIGHT	CLAVICLE_RIGHT
13	ELBOW_RIGHT	SHOULDER_RIGHT
14	WRIST_RIGHT	ELBOW_RIGHT
'''
def init():
    lib.init()

def get_data():
    # ctypes._reset_cache()
    data=lib.get_data()
    l_shoulder_pos  = data[0]
    r_shoulder_pos  = data[1]
    l_elbow_pos     = data[2]
    r_elbow_pos     = data[3]
    l_wrist_pos     = data[4]
    r_wrist_pos     = data[5]
    # l_shoulder_pos = lib.get_data(5)
    # r_shoulder_pos = lib.get_data(12)
    # l_elbow_pos = lib.get_data(6)
    # r_elbow_pos = lib.get_data(13)
    # l_wrist_pos = lib.get_data(7)
    # r_wrist_pos = lib.get_data(14)
    pos=[]
    pos.append([l_shoulder_pos[0], l_shoulder_pos[1], l_shoulder_pos[2]])
    pos.append([r_shoulder_pos[0], r_shoulder_pos[1], r_shoulder_pos[2]])
    pos.append([l_elbow_pos[0], l_elbow_pos[1], l_elbow_pos[2]])
    pos.append([r_elbow_pos[0], r_elbow_pos[1], r_elbow_pos[2]])
    pos.append([l_wrist_pos[0], l_wrist_pos[1], l_wrist_pos[2]])
    pos.append([r_wrist_pos[0], r_wrist_pos[1], r_wrist_pos[2]])
    quat=[]
    quat.append([l_shoulder_pos[3], l_shoulder_pos[4], l_shoulder_pos[5], l_shoulder_pos[6]])
    quat.append([r_shoulder_pos[3], r_shoulder_pos[4], r_shoulder_pos[5], l_shoulder_pos[6]])
    quat.append([l_elbow_pos[3], l_elbow_pos[4], l_elbow_pos[5], l_shoulder_pos[6]])
    quat.append([r_elbow_pos[3], r_elbow_pos[4], r_elbow_pos[5], l_shoulder_pos[6]])
    quat.append([l_wrist_pos[3], l_wrist_pos[4], l_wrist_pos[5], l_shoulder_pos[6]])
    quat.append([r_wrist_pos[3], r_wrist_pos[4], r_wrist_pos[5], l_shoulder_pos[6]])
    lib.free_memory()
    return pos,quat

def deinit():
    lib.deinit()


# init()
# while True:
#     a,b=get_data()
#     # time.sleep(0.1)
#     # print("*",end="")
# deinit()
