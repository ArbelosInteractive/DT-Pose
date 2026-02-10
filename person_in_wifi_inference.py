import os
import argparse
import math

import yaml
import numpy as np
import torch

from feeder.person_in_wifi_3d import PersonInWif3D, piw3_make_dataloader

from utils import *
from model.model import *
from utils import setup_seed

from model.metafi.mynetwork import metafinet, metafi_weights_init
from model.hpeli.hpeli import hpelinet, hpeli_weights_init

from render_pose import render_pose_3d

import matplotlib.pyplot as plt


BONES = [
    (0, 1), (1, 2), (2, 3),        # torso
    (2, 4), (4, 5), (5, 6),        # left arm
    (2, 7), (7, 8), (8, 9),        # right arm
    (0,10), (10,11),               # left leg
    (0,12), (12,13),               # right leg
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pose Decoding Stage")
    parser.add_argument("--config_file", type=str, help="Configuration YAML file", default='config/person_in_wifi_3d/pose_config.yaml')
    parser.add_argument("--model", type=str, help="Pose Model File", default='pose_weights/person-in-wifi-3d/one-person/pose_pretrain/DT-Pose/pose_mpjpe.pt')
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    
    setup_seed(config['seed'])
    dataset_root = config['dataset_root']

    batch_size = config['batch_size']
    load_batch_size = min(config['max_device_batch_size'], batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    if config['dataset_name'] == 'person-in-wifi-3d':
        train_dataset = PersonInWif3D('training', dataset_root, config['setting'])
        val_dataset = PersonInWif3D('validation', dataset_root, config['setting'])
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = piw3_make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size)
        val_loader = piw3_make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=load_batch_size)
    else:
        print('No dataset!')

    # TODO: Settings, e.g., your model, optimizer, device, ...
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config['pretrained_model_path'] is not None:
        print('*'*20+'   Load Model Weights   '+'*'*20)
        print('*'*20+'  '+config['dataset_name']+','+config['setting']+','+config['experiment_name']+'   '+'*'*20)
        model = torch.load(args.model, map_location=device, weights_only=False)
        model = model.to(device)
        # writer = SummaryWriter(os.path.join('logs', config['dataset_name'], config['setting'], 'pose_pretrain',config['experiment_name']))
       
        # if config['dataset_name'] == 'person-in-wifi-3d':
        #     model = ViT_Pose_Decoder(model.encoder, keypoints=14, coor_num=3, token_num=90*10, dataset=config['dataset_name'], num_person=config['num_person']).to(device)
        #     optim = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3) 

        # save test feature
        test_feature_path = os.path.join('features', config['dataset_name'], config['setting'], 'pose_pretrain', config['experiment_name'])
        if not os.path.exists(test_feature_path):
            os.makedirs(test_feature_path)

    model.eval()
    gt_list = []
    pred_list = []
    feature_list = []
    wifi_list = []
    label_list = []
    attention_list_first = []
    attention_list_second = []
    with torch.no_grad():
        losses = []
        mpjpe_list = []
        pampjpe_list = []
        mpjpe_joints_list = []
        pampjpe_joints_list = []
        mpjpe_align_list = []
        pampjpe_align_list = []
        
        pck_iter = [[] for _ in range(5)]
        pck_align_iter = [[] for _ in range(5)]
        subject_mpjpe = {}
        for batch_idx, batch_data in enumerate(val_loader):
            val_csi = batch_data['input_wifi-csi'].to(device)
            val_pose_gt = batch_data['output'].to(device)
            # if config['dataset_name'] == 'mmfi-csi' or config['dataset_name'] == 'wipose':
            #     label = batch_data['label'].to(device)
            #     label_list.append(label.data.cpu().numpy())

            predicted_val_pose, pred_fea = model(val_csi)

            if config['dataset_name'] == 'person-in-wifi-3d':
                person_num = batch_data['person_num'].to(device)
                predicted_val_pose = torch.cat([predicted_val_pose[:,:num,:,:].reshape((-1, 14, 3)) for num in person_num], dim=0)
                val_pose_gt = torch.cat([val_pose_gt[:,:num,:,:].reshape((-1, 14, 3)) for num in person_num], dim=0)


            feature_list.append(pred_fea.data.cpu().numpy())
            wifi_list.append(val_csi.data.cpu().numpy())
            gt_list.append(val_pose_gt.data.cpu().numpy())
            pred_list.append(predicted_val_pose.data.cpu().numpy())
            
            
            loss = torch.mean(torch.norm(predicted_val_pose-val_pose_gt, dim=-1))
            print(loss.item())
            # calculate the pck, mpjpe, pampjpe
            for idx, percentage in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                pck_iter[idx].append(compute_pck_pckh(predicted_val_pose.permute(0,2,1).data.cpu().numpy(), val_pose_gt.permute(0,2,1).data.cpu().numpy(), percentage, align=False, dataset=config['dataset_name']))
            mpjpe, pampjpe, mpjpe_joints, pampjpe_joints = calulate_error(predicted_val_pose.data.cpu().numpy(), val_pose_gt.data.cpu().numpy(), align=False)
            mpjpe_list += mpjpe.tolist()
            pampjpe_list += pampjpe.tolist()
            losses.append(loss.item())
        avg_val_loss = sum(losses) / len(losses)
        avg_val_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
        avg_val_pampjpe = sum(pampjpe_list) / len(pampjpe_list)
        if config['dataset_name'] == 'mmfi-csi':
            pck_overall = [np.mean(pck_value, 0)[17] for pck_value in pck_iter]
        elif config['dataset_name'] == 'person-in-wifi-3d':
            pck_overall = [np.mean(pck_value, 0)[14] for pck_value in pck_iter]
        elif config['dataset_name'] == 'wipose':
            pck_overall = [np.mean(pck_value, 0)[18] for pck_value in pck_iter]
            
        gt_list = np.concatenate(gt_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        feature_list = np.concatenate(feature_list, axis=0)
        wifi_list = np.concatenate(wifi_list, axis=0)

        # print(pck_overall)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection="3d")

        num_frames = pred_list.shape[0]
        frame_idx = 0
        while True:
            print(f"Frame {frame_idx+1}/{num_frames}")

            ax.cla()   # clear previous frame
            ax2.cla()  # clear previous frame

            render_pose_3d(gt_list[frame_idx], ax=ax)
            render_pose_3d(pred_list[frame_idx], ax=ax2)

            plt.pause(0.001)  # allow the figure to update

            command = input("Press Enter for next, b for back, q to quit: ").strip().lower()
            if command == "q":
                break
            elif command == "b":
                frame_idx = max(0, frame_idx - 1)
            else:
                frame_idx = min(num_frames - 1, frame_idx + 1)
        if config['dataset_name'] == 'mmfi-csi' or config['dataset_name'] == 'wipose':
            label_list = np.concatenate(label_list, axis=0)
        
    # print(f'In epoch {epoch}, test losss: {avg_val_loss}')
    print(f'test mpjpe: {avg_val_mpjpe}, test pa-mpjpe: {avg_val_pampjpe}, test pck50: {pck_overall[0]}, test pck40: {pck_overall[1]}, test pck30: {pck_overall[2]}, test pck20: {pck_overall[3]}, test pck10: {pck_overall[4]}.')
 
    # if config['dataset_name'] == 'person-in-wifi-3d':
    #     np.savez(os.path.join(test_feature_path, 'test_feature.npz'), fea=feature_list, pred_pose=pred_list, gt_pose=gt_list, att=[attention_list_first, attention_list_second], wifi=wifi_list)    
  
