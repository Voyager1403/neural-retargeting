import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as transforms
from torch_geometric.data import Data, Batch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import h5py
import argparse
import logging
import time
import os
import copy
from datetime import datetime

import dataset
from dataset import Normalize, parse_h5
from models import model
from models.loss import CollisionLoss, JointLimitLoss, RegLoss
from train import train_epoch
from utils.config import cfg
from utils.util import create_folder

# Argument parse
parser = argparse.ArgumentParser(description='Inference with trained model')
parser.add_argument('--cfg', default='configs/inference/yumi.yaml', type=str, help='Path to configuration file')
args = parser.parse_args()

# Configurations parse
cfg.merge_from_file(args.cfg)
cfg.freeze()
print(cfg)

# Create folder
create_folder(cfg.OTHERS.SAVE)
create_folder(cfg.OTHERS.LOG)
create_folder(cfg.OTHERS.SUMMARY)

# Create logger & tensorboard writer
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(os.path.join(cfg.OTHERS.LOG, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
logger = logging.getLogger()
writer = SummaryWriter(os.path.join(cfg.OTHERS.SUMMARY, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Load data
    pre_transform = transforms.Compose([Normalize()])
    test_data, l_hand_angle, r_hand_angle= parse_h5(filename=cfg.INFERENCE.MOTION.SOURCE, selected_key=cfg.INFERENCE.MOTION.KEY)
    test_data = [pre_transform(data) for data in test_data]
    indices = [idx for idx in range(0, len(test_data), cfg.HYPER.BATCH_SIZE)]
    test_loader = [test_data[idx: idx+cfg.HYPER.BATCH_SIZE] for idx in indices]
    test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target : target.skeleton_type)
    hf = h5py.File(os.path.join(cfg.INFERENCE.H5.PATH, 'source.h5'), 'w')
    g1 = hf.create_group('group1')
    source_pos = torch.stack([data.pos for data in test_data], dim=0)
    g1.create_dataset('l_joint_pos_2', data=source_pos[:, :3])
    g1.create_dataset('r_joint_pos_2', data=source_pos[:, 3:])
    hf.close()
    print('Source H5 file saved!')

    # Create model
    model = getattr(model, cfg.MODEL.NAME)().to(device)

    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

    # store initial z
    model.eval()
    z_all = []
    for batch_idx, data_list in enumerate(test_loader):
        for target_idx, target in enumerate(test_target):
            # fetch target
            target_list = [target for data in data_list]
            # forward
            z = model.encode(Batch.from_data_list(data_list).to(device)).detach()
#             z = torch.empty(Batch.from_data_list(target_list).x.size(0), 64).normal_(mean=0, std=0.005).to(device)
            z.requires_grad = True
            z_all.append(z)

    # Create loss criterion
    # end effector loss
    ee_criterion = nn.MSELoss() if cfg.LOSS.EE else None
    # vector similarity loss
    vec_criterion = nn.MSELoss() if cfg.LOSS.VEC else None
    # collision loss
    col_criterion = CollisionLoss(cfg.LOSS.COL_THRESHOLD) if cfg.LOSS.COL else None
    # joint limit loss
    lim_criterion = JointLimitLoss() if cfg.LOSS.LIM else None
    # end effector orientation loss
    ori_criterion = nn.MSELoss() if cfg.LOSS.ORI else None
    # regularization loss
    reg_criterion = RegLoss() if cfg.LOSS.REG else None

    # Create optimizer
    optimizer = optim.Adam(z_all, lr=cfg.HYPER.LEARNING_RATE)

    best_loss = float('Inf')
    best_z_all = copy.deepcopy(z_all)
    best_cnt = 0
    start_time = time.time()

    # latent optimization
    for epoch in range(cfg.HYPER.EPOCHS):
        train_loss = train_epoch(model,
                                 ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, reg_criterion,
                                 optimizer,
                                 test_loader, test_target,
                                 epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device, z_all)
        # Save model
        if train_loss > best_loss:
            best_cnt += 1
        else:
            best_cnt = 0
            best_loss = train_loss
            best_z_all = copy.deepcopy(z_all)
        if best_cnt == 5:
            logger.info("Interation Finished")
            break
        print(best_cnt)

    # store final results
    model.eval()
    pos_all = []
    ang_all = []
    for batch_idx, data_list in enumerate(test_loader):
        for target_idx, target in enumerate(test_target):
            # fetch target
            target_list = [target for data in data_list]
            # fetch z
            z = best_z_all[batch_idx]
            # forward
            target_ang, target_pos, _, _, _, _, target_global_pos = model.decode(z, Batch.from_data_list(target_list).to(z.device))
            pos_all.append(target_global_pos)
            ang_all.append(target_ang)
    
    if cfg.INFERENCE.H5.BOOL:
        pos = torch.cat(pos_all, dim=0).view(len(test_data), -1, 3).detach().cpu().numpy() # [T, joint_num, xyz]
        ang = torch.cat(ang_all, dim=0).view(len(test_data), -1).detach().cpu().numpy()
        hf = h5py.File(os.path.join(cfg.INFERENCE.H5.PATH, 'inference.h5'), 'w')
        g1 = hf.create_group('group1')
        g1.create_dataset('l_joint_pos_2', data=pos[:, :7])
        g1.create_dataset('r_joint_pos_2', data=pos[:, 7:])
        g1.create_dataset('l_joint_angle_2', data=ang[:, :7])
        g1.create_dataset('r_joint_angle_2', data=ang[:, 7:])
        g1.create_dataset('l_glove_angle_2', data=l_hand_angle)
        g1.create_dataset('r_glove_angle_2', data=r_hand_angle)
        hf.close()
        print('Target H5 file saved!')
