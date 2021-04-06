import torch
from torch_geometric.data import Data, Batch
from models.loss import calculate_ee_loss, calculate_vec_loss, calculate_lim_loss, calculate_ori_loss
import time

def test_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, reg_criterion, dataloader, target_skeleton, epoch, logger, log_interval, writer, device):
    logger.info("Testing Epoch {}".format(epoch+1).center(60, '-'))
    start_time = time.time()

    model.eval()
    ee_losses = []
    vec_losses = []
    col_losses = []
    lim_losses = []
    ori_losses = []
    reg_losses = []

    with torch.no_grad():
        for batch_idx, data_list in enumerate(dataloader):
            for target_idx, target in enumerate(target_skeleton):
                # fetch target
                target_list = [target for data in data_list]

                # forward
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

    # Compute average loss
    ee_loss = sum(ee_losses)/len(ee_losses)
    vec_loss = sum(vec_losses)/len(vec_losses)
    col_loss = sum(col_losses)/len(col_losses)
    lim_loss = sum(lim_losses)/len(lim_losses)
    ori_loss = sum(ori_losses)/len(ori_losses)
    reg_loss = sum(reg_losses)/len(reg_losses)
    test_loss = ee_loss + vec_loss + col_loss + lim_loss + ori_loss + reg_loss
    # Log
    writer.add_scalars('testing_loss', {'test': test_loss}, epoch+1)
    writer.add_scalars('end_effector_loss', {'test': ee_loss}, epoch+1)
    writer.add_scalars('vector_loss', {'test': vec_loss}, epoch+1)
    writer.add_scalars('collision_loss', {'test': col_loss}, epoch+1)
    writer.add_scalars('joint_limit_loss', {'test': lim_loss}, epoch+1)
    writer.add_scalars('orientation_loss', {'test': ori_loss}, epoch+1)
    writer.add_scalars('regularization_loss', {'test': reg_loss}, epoch+1)
    end_time = time.time()
    logger.info("Epoch {:04d} | Testing Time {:.2f} s | Avg Testing Loss {:.6f} | Avg EE Loss {:.6f} | Avg Vec Loss {:.6f} | Avg Col Loss {:.6f} | Avg Lim Loss {:.6f} | Avg Ori Loss {:.6f} | Avg Reg Loss {:.6f}".format(epoch+1, end_time-start_time, test_loss, ee_loss, vec_loss, col_loss, lim_loss, ori_loss, reg_loss))

    return test_loss
