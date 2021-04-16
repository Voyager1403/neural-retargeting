import torch
import torch.nn as nn
import torch_geometric.transforms as transforms

import argparse
# import os

from models import model as model1
from models.parallel import DataParallel
# from models.loss import CollisionLoss, JointLimitLoss
import dataset
#from dataset import Degree2Radian, Normalize, parse_bvh_to_frame, parse_bvh_to_motion, parse_bvh_to_hand, parse_h5, parse_h5_temporal
from dataset import Degree2Radian, Normalize, parse_h5
from parse_from_Kinect import parse_from_Kinect
from utils.config import cfg
from utils.util import create_folder


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
        self.model = getattr(model1, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.DIM).to(inference.device)

        # Run the model parallelly
        print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
        self.model = DataParallel(self.model).to(inference.device)

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
            data_list = [inference.pre_transform(data).to(inference.device) for data in data_list]
            target = inference.targets[cfg.INFERENCE.TARGET]
            target_list = []
            for data in data_list:
                target_list.append(target)

            # forward
            ang, _, _, _, _, rot, pos = self.model(data_list, target_list)

            # reshape
            ang = ang.view(sum([data.t if hasattr(data, 't') else 1 for data in target_list]), -1).cpu().numpy() # [T, joint_num]
            pos = pos.view(sum([data.t if hasattr(data, 't') else 1 for data in target_list]), -1, 3) # [T, joint_num, xyz]

            #output
            l_joint_angle_2=ang[:, :7]
            r_joint_angle_2=ang[:, 7:]
            l_glove_angle_2=l_hand_angle
            r_glove_angle_2=r_hand_angle
            return l_joint_angle_2,r_joint_angle_2,l_glove_angle_2,r_glove_angle_2
