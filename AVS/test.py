import torch
import torch.nn as nn
import dataloader as dataloader
import models
import model.Swin_AVSModel as AVSModel
import model.Swin_AVSModel_Base as AVSModelBase
import model.Swin_AVSModel_without_adapt as AVSModelWithoutAdapt
import numpy as np
import warnings
import json
from sklearn import metrics
from tqdm import tqdm
import clip
import random
from loss import IouSemanticAwareLoss, mask_iou
from utilities import *
import time
import os
from PIL import Image

# all exp in this work is based on 224 * 224 image
im_res = 224

dir_image = './STG-CMA/AVS/dataset/Single-source/s4_data/visual_frames'
dir_audio_log_mel = './STG-CMA/AVS/dataset/Single-source/s4_data/audio_log_mel'
dir_audio_wav = './STG-CMA/AVS/dataset/Single-source/s4_data/audio_wav'
dir_mask = './STG-CMA/AVS/dataset/Single-source/s4_data/gt_masks'
dataset_mean = -5.669627666473389
dataset_std = 3.948380470275879
batch_size = 2
num_workers = 16
pretrain_path = None
ftmode = 'fusion'
ckpt_path = './STG-CMA/AVS/exp/ave_29_ft-bal-MM-Swin-AVS-Large-3e-4-10-0.7-1-bs2-ldaFalse-fusion-fzTrue-h0.1-a5-Swin-Large-AVS-Dec25-AddTemporal-WithoutAdapt/models/best_audio_model.pth'
#'./STG-CMA/AVS/exp/ave_29_ft-bal-MM-Swin-AVS-Large-3e-4-10-0.7-1-bs2-ldaFalse-fusion-fzTrue-h0.1-a5-Swin-Large-AVS-Dec25-WithoutAdapt/models/best_audio_model.pth'
#'./STG-CMA/AVS/exp/ave_29_ft-bal-MM-Swin-AVS-Large-3e-4-10-0.7-1-bs2-ldaFalse-fusion-fzTrue-h0.1-a5-Swin-Base-AVS-Nov28-New/models/best_audio_model.pth'
mask_save_path = './STG-CMA/AVS/save_pred_masks_withoutadapt_latest'
#'./STG-CMA/AVS/save_pred_masks_withoutadapt'
#'./STG-CMA/AVS/save_pred_masks'

def save_mask(pred_masks, save_base_path, category_list, video_name_list):
    # pred_mask: [bs*5, 1, 224, 224]
    # print(f"=> {len(video_name_list)} videos in this batch")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    pred_masks = pred_masks.squeeze(2)
    pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()

    pred_masks = pred_masks.view(-1, 5, pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    bs = pred_masks.shape[0]

    for idx in range(bs):
        category, video_name = category_list[idx], video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, category, video_name)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_masks[idx]  # [5, 1, 224, 224]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            output_name = "%s_%d.png" % (video_name, video_id)
            im = Image.fromarray(one_mask).convert('P')
            im.save(os.path.join(mask_save_path, output_name), format='PNG')

def validate(audio_model, val_loader, ftmode, mask_save_path, save_pred_mask=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    miou_meter = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    # A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (imgs, audio_spec, audio, mask, category_list, video_name_list) in enumerate(val_loader):
            imgs = imgs.to(device)  # [b, 5, 3, 224, 224]
            audio_spec = audio_spec.to(device)  # [b, 5, 224, 224]
            mask = mask.to(device)  # [b, 5, 1, 224, 224]
            B, frame, C, H, W = imgs.shape
            mask = mask.view(B * frame, H, W)  # [bf, 224, 224]

            # with autocast():
            output, _, _ = audio_model(audio_spec, imgs, ftmode)  # [bf, 1, 224, 224]
            if save_pred_mask:
                save_mask(output.squeeze(1), mask_save_path, category_list, video_name_list)
            miou = mask_iou(output.squeeze(1), mask)  # [bf, 224, 224] , mask: [bf, 224, 224]
            # print(miou)
            # todo
            miou_meter.update(miou.item())

            batch_time.update(time.time() - end)
            end = time.time()

        miou_score = miou_meter.avg


    return miou_score

# test loader
val_audio_conf = {'mode': 'test', 'dir_image': dir_image, 'dir_audio_log_mel': dir_audio_log_mel,
                  'dir_audio_wav': dir_audio_wav, 'dir_mask': dir_mask, 'mean': dataset_mean,
                  'std': dataset_std}  # , 'mixup': 0

val_loader = torch.utils.data.DataLoader(
    dataloader.S4Dataset(audio_conf=val_audio_conf),
    batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# model
'''
audio_model = AVSModel.SwinTransformer2D_Adapter_AVS(
                                                   patch_size=[1, 4, 4],
                                                   img_size=224,
                                                   num_frames=5,
                                                   embed_dim=192,
                                                   depths=[2, 2, 18, 2],
                                                   num_heads=[6, 12, 24, 48],
                                                   window_size=7,
                                                   pretrained=pretrain_path,
                                                   ftmode=ftmode,
                                                   adapter_mlp_ratio=[0.5, 0.25, 0.125, 0.0625],
                                                   channel=256,
                                                   opt=None, config=None, vis_dim=[64, 128, 320, 512],
                                                   tpavi_stages=[0, 1, 2, 3],
                                                   tpavi_vv_flag=False,
                                                   tpavi_va_flag=True)'''

audio_model = AVSModelWithoutAdapt.SwinTransformer2D_Adapter_AVS_Without_Adapt(
        patch_size=[1, 4, 4],
        img_size=224,
        num_frames=5,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        pretrained=pretrain_path,
        ftmode=ftmode,
        adapter_mlp_ratio=[0.5, 0.25, 0.125, 0.0625],
        channel=256,
        opt=None, config=None, vis_dim=[64, 128, 320, 512],
        tpavi_stages=[0, 1, 2, 3],
        tpavi_vv_flag=False,
        tpavi_va_flag=True
    )
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

# load pre-trained model
ckpt_model = torch.load(ckpt_path, map_location='cpu')
print('now load checkpoints from', ckpt_path)
msg = audio_model.load_state_dict(ckpt_model, strict=True)
print(msg)
miou_score = validate(audio_model, val_loader, ftmode, mask_save_path, save_pred_mask=False)
print(miou_score)
