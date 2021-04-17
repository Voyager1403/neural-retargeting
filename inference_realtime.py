import torch
import torch.nn as nn
import torch_geometric.transforms as transforms

import argparse
# import os
from torch_geometric.data import Data, Batch
from models import model as model1
from models.parallel import DataParallel
# from models.loss import CollisionLoss, JointLimitLoss
import dataset
#from dataset import Degree2Radian, Normalize, parse_bvh_to_frame, parse_bvh_to_motion, parse_bvh_to_hand, parse_h5, parse_h5_temporal
from dataset import Normalize, parse_h5
from parse_from_Kinect import parse_from_Kinect
from utils.config import cfg
from utils.util import create_folder
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
# Create logger & tensorboard writer
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(os.path.join(cfg.OTHERS.LOG, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
logger = logging.getLogger()
writer = SummaryWriter(os.path.join(cfg.OTHERS.SUMMARY, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))
class inference:
    # Argument parse
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--cfg', default='configs/inference/yumi.yaml', type=str, help='Path to configuration file')
    args = parser.parse_args()

    # Configurations parse
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)
    pre_transform = transforms.Compose([Normalize()])
    targets = sorted(
        [target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)],
        key=lambda target: target.skeleton_type)
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):


        # create folder
        create_folder(cfg.INFERENCE.SAVE)

        # Load data
        transform = None


        # Create model
        self.model = getattr(model1, cfg.MODEL.NAME)().to(self.device)

        # Run the model parallelly
        print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
        # self.model = DataParallel(self.model).to(inference.device)

        # Load checkpoint
        if cfg.MODEL.CHECKPOINT is not None:
            self.model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

        # Create loss criterion
        # reconstruction loss
        rec_criterion = nn.MSELoss()
        # end effector loss
        ee_criterion = nn.MSELoss()
    
    def step(self,pos,quat):
        # one step Inference
        with torch.no_grad():

            # get data from kinect
            data, l_hand_angle, r_hand_angle = parse_from_Kinect(pos,quat,filename=cfg.INFERENCE.MOTION.SOURCE)

            # process of data
            data_list = data
            indices = [idx for idx in range(0, len(data_list), 1)]
            data_loader = [data_list[idx: idx + cfg.HYPER.BATCH_SIZE] for idx in indices]


            # target = inference.targets[cfg.INFERENCE.TARGET]
            target = inference.targets[0]
            target_list = []
            for data in data_list:
                target_list.append(target)
            #
            # # forward
            # ang, _, _, _, _, rot, pos = self.model(data_list, target_list)


            #latent space
            # store initial z
            self.model.eval()
            z_all = []
            for batch_idx, data_list in enumerate(data_loader):
                for target_idx, target in enumerate(target_list):
                    # fetch target
                    target_list = [target for data in data_list]
                    # forward
                    z = self.model.encode(Batch.from_data_list(data_list).to(self.device)).detach()
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
                train_loss,ang = train_epoch(self.model, ee_criterion,  vec_criterion, col_criterion, lim_criterion,
                                         ori_criterion, reg_criterion, optimizer, data_loader, target_list, epoch,
                                         logger, cfg.OTHERS.LOG_INTERVAL, writer, self.device, z_all)
                # Save model
                if train_loss > best_loss:
                    best_cnt += 1
                else:
                    best_cnt = 0
                    best_loss = train_loss
                    best_z_all = copy.deepcopy(z_all)
                if best_cnt == 5:
                    # logger.info("Interation Finished")
                    print("Interation Finished")
                    break
                print(best_cnt)

            # store final results
            self.model.eval()
            pos_all = []
            ang_all = []
            for batch_idx, data_list in enumerate(data_loader):
                for target_idx, target in enumerate(target_list):
                    # fetch target
                    target_list = [target for data in data_list]
                    # fetch z
                    z = best_z_all[batch_idx]
                    # forward
                    target_ang, target_pos, _, _, _, _, target_global_pos = self.model.decode(z, Batch.from_data_list(
                        target_list).to(z.device))
                    pos_all.append(target_global_pos)
                    ang_all.append(target_ang)


            # reshape
            # ang = ang.view(sum([data.t if hasattr(data, 't') else 1 for data in target_list]), -1).cpu().numpy() # [T, joint_num]
            # pos = pos.view(sum([data.t if hasattr(data, 't') else 1 for data in target_list]), -1, 3) # [T, joint_num, xyz]
            pos = torch.cat(pos_all, dim=0).view(len(data_loader), -1, 3).detach().cpu().numpy()  # [T, joint_num, xyz]
            ang = torch.cat(ang_all, dim=0).view(len(target_list), -1).detach().cpu().numpy()

            #output
            l_joint_angle_2=ang[:, :7]
            r_joint_angle_2=ang[:, 7:]
            l_glove_angle_2=l_hand_angle
            r_glove_angle_2=r_hand_angle
            return l_joint_angle_2,r_joint_angle_2,l_glove_angle_2,r_glove_angle_2
