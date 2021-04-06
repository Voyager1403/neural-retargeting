import torch
from torch_geometric.data import Data, Batch
from models.loss import calculate_ee_loss, calculate_vec_loss, calculate_lim_loss, calculate_ori_loss
import time

def train_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, reg_criterion, optimizer, dataloader, target_skeleton, epoch, logger, log_interval, writer, device, z_all=None, ang_all=None):
    logger.info("Training Epoch {}".format(epoch+1).center(60, '-'))
    start_time = time.time()

    model.train()
    ee_losses = []
    vec_losses = []
    col_losses = []
    lim_losses = []
    ori_losses = []
    reg_losses = []

    for batch_idx, data_list in enumerate(dataloader):
        for target_idx, target in enumerate(target_skeleton):
            # zero gradient
            optimizer.zero_grad()

            # fetch target
            target_list = [target for data in data_list]

            # forward
            if z_all is not None:
                z = z_all[batch_idx]
                target_ang, target_pos, _, _, _, target_rot, target_global_pos = model.decode(z, Batch.from_data_list(target_list).to(device))
            elif ang_all is not None:
                ang = ang_all[batch_idx]
                target_ang, target_pos, _, _, _, target_rot, target_global_pos = model.forward_kinematics(ang, Batch.from_data_list(target_list).to(device))
            else:
                target_ang, target_pos, z, cycle_z, cycle_x, target_rot, target_global_pos = model(Batch.from_data_list(data_list).to(device), Batch.from_data_list(target_list).to(device))

            # end effector loss
            if ee_criterion:
                ee_loss = calculate_ee_loss(data_list, target_list, target_pos, ee_criterion)*1000
                ee_losses.append(ee_loss.item())
            else:
                ee_loss = 0
                ee_losses.append(0)

            # vector loss
            if vec_criterion:
                vec_loss = calculate_vec_loss(data_list, target_list, target_pos, vec_criterion)*100
                vec_losses.append(vec_loss.item())
            else:
                vec_loss = 0
                vec_losses.append(0)

            # collision loss
            if col_criterion:
                col_loss = col_criterion(target_global_pos.view(sum([data.t if hasattr(data, 't') else 1 for data in target_list]), -1, 3), target.edge_index,
                                         target_rot.view(sum([data.t if hasattr(data, 't') else 1 for data in target_list]), -1, 9), target.ee_mask)*100
                col_losses.append(col_loss.item())
            else:
                col_loss = 0
                col_losses.append(0)

            # joint limit loss
            if lim_criterion:
                lim_loss = calculate_lim_loss(target_list, target_ang, lim_criterion)*10000
                lim_losses.append(lim_loss.item())
            else:
                lim_loss = 0
                lim_losses.append(0)
            
            # end effector orientation loss
            if ori_criterion:
                ori_loss = calculate_ori_loss(data_list, target_list, target_rot, ori_criterion)*100
                ori_losses.append(ori_loss.item())
            else:
                ori_loss = 0
                ori_losses.append(0)
            
            # regularization loss
            if reg_criterion:
                reg_loss = reg_criterion(z.view(sum([data.t if hasattr(data, 't') else 1 for data in target_list]), -1, 64))
                reg_losses.append(reg_loss.item())
            else:
                reg_loss = 0
                reg_losses.append(0)

            # backward
            loss = ee_loss + vec_loss + col_loss + lim_loss + ori_loss + reg_loss
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            # optimize
            optimizer.step()

        # log
        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:04d} | iteration {:05d} | EE {:.6f} | Vec {:.6f} | Col {:.6f} | Lim {:.6f} | Ori {:.6f} | Reg {:.6f}".format(epoch+1, batch_idx+1, ee_losses[-1], vec_losses[-1], col_losses[-1], lim_losses[-1], ori_losses[-1], reg_losses[-1]))

    # Compute average loss
    ee_loss = sum(ee_losses)/len(ee_losses)
    vec_loss = sum(vec_losses)/len(vec_losses)
    col_loss = sum(col_losses)/len(col_losses)
    lim_loss = sum(lim_losses)/len(lim_losses)
    ori_loss = sum(ori_losses)/len(ori_losses)
    reg_loss = sum(reg_losses)/len(reg_losses)
    train_loss = ee_loss + vec_loss + col_loss + lim_loss + ori_loss + reg_loss
    # Log
    writer.add_scalars('training_loss', {'train': train_loss}, epoch+1)
    writer.add_scalars('end_effector_loss', {'train': ee_loss}, epoch+1)
    writer.add_scalars('vector_loss', {'train': vec_loss}, epoch+1)
    writer.add_scalars('collision_loss', {'train': col_loss}, epoch+1)
    writer.add_scalars('joint_limit_loss', {'train': lim_loss}, epoch+1)
    writer.add_scalars('orientation_loss', {'train': ori_loss}, epoch+1)
    writer.add_scalars('regularization_loss', {'train': reg_loss}, epoch+1)
    end_time = time.time()
    logger.info("Epoch {:04d} | Training Time {:.2f} s | Avg Training Loss {:.6f} | Avg EE Loss {:.6f} | Avg Vec Loss {:.6f} | Avg Col Loss {:.6f} | Avg Lim Loss {:.6f} | Avg Ori Loss {:.6f} | Avg Reg Loss {:.6f}".format(epoch+1, end_time-start_time, train_loss, ee_loss, vec_loss, col_loss, lim_loss, ori_loss, reg_loss))

    return train_loss
