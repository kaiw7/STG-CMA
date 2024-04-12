import argparse
import os

os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler

sys.path.append('./AudioVisual')
# basepath = os.path.dirname(os.path.dirname(sys.path[0]))
# sys.path.append(basepath)
# import dataloader_new as dataloader
import dataloader as dataloader
# import dataloader_new as dataloader
import models
import model.Swin_AVSModel as AVSModel
import model.Swin_AVSModel_Base as AVSModelBase
import model.Swin_AVSModel_without_adapt as AVSModelWithoutAdapt
import numpy as np
import warnings
import json
from sklearn import metrics
from traintest_adapt_avs import train, validate  # todo our final setting
# from traintest_adapt_ave29_ft import train, validate # todo for full finetuning
from tqdm import tqdm
import clip
import random

# finetune cav-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used",
                    choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400", 'ks-32',
                             'ucf101', 'hmdb51', 'ave-29', 'avsbench'])
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization in audio branch")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization in audio branch")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW',
                    help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=10, help="number of maximum training epochs")
# not used in the formal experiments, only in preliminary experiments
parser.add_argument("--lr_patience", type=int, default=1,
                    help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning",
                    choices=["mAP", "acc", "miou"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task",
                    choices=["BCE", "CE", "IoU"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=2, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=1, type=int,
                    help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")

# todo newly added
parser.add_argument("--finetune_path", type=str, default='None', help="finetune_path model path")

parser.add_argument("--head_lr", type=float, default=50.0,
                    help="learning rate ratio the newly initialized layers / pretrained weights")
parser.add_argument('--freeze_base', help='freeze the backbone or not', type=ast.literal_eval)
parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval)

# todo newly added para for cosine schedule
parser.add_argument("--lr_cosine_adapt", help='if use cosine adaptive learning rate', type=ast.literal_eval)
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')

# todo newly added para for AVS
parser.add_argument("--dir_image", type=str, default='', help="training data json")
parser.add_argument("--dir_audio_log_mel", type=str, default='', help="validation data json")
parser.add_argument("--dir_audio_wav", type=str, default='', help="training data json")
parser.add_argument("--dir_mask", type=str, default='', help="validation data json")

args = parser.parse_args()
# finalized seed=0
seed = 0  # 1888 #1667 #1888 #1777 #1666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# all exp in this work is based on 224 * 224 image
im_res = 224

audio_conf = {'mode': 'train', 'dir_image': args.dir_image, 'dir_audio_log_mel': args.dir_audio_log_mel,
              'dir_audio_wav': args.dir_audio_wav, 'dir_mask': args.dir_mask, 'mean': args.dataset_mean,
              'std': args.dataset_std}  # , 'mixup': args.mixup

val_audio_conf = {'mode': 'test', 'dir_image': args.dir_image, 'dir_audio_log_mel': args.dir_audio_log_mel,
                  'dir_audio_wav': args.dir_audio_wav, 'dir_mask': args.dir_mask, 'mean': args.dataset_mean,
                  'std': args.dataset_std}  # , 'mixup': 0
'''
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
              'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res, 'ftmode': args.ftmode,
              'model_tpye':args.model}
'''

'''
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res, 'ftmode': args.ftmode,
                  'model_tpye':args.model}
'''

if args.bal == 'bal':
    print('balanced sampler is being used')
    if args.weight_file == None:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_weight.csv', delimiter=',')
    else:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.S4Dataset(audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.S4Dataset(audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.S4Dataset(audio_conf=val_audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.S4Dataset(audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if args.pretrain_path == 'None':
    args.pretrain_path = None

if args.model == 'cav-mae-ft':
    print('finetune a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
    audio_model = models.CAVMAEFT_ADAPT(label_dim=args.n_class, modality_specific_depth=11, num_video_frames=10)
elif args.model == 'ViT':
    print('finetune ViT 12 modality-specific layers')
    audio_model = models.ViT_ImageNet(label_dim=args.n_class, depth=12, num_video_frames=10,
                                      pretrained=args.pretrain_path)
elif args.model == 'CLIP':
    print('finetune CLIP 12 modality-specific layers')
    audio_model = models.ViT_CLIP(label_dim=args.n_class, layers=12, num_video_frames=10,
                                  pretrained=args.pretrain_path)
elif args.model == 'MM-ViT':
    print('finetune MM-ViT 12 modality-specific layers')
    audio_model = models.MM_ViT_ImageNet(label_dim=args.n_class, depth=12, fusion_depth=1, num_video_frames=10,
                                         pretrained=args.pretrain_path, mode=args.ftmode)
elif args.model == 'MM-DeiT':
    print('finetune MM-DeiT 12 modality-specific layers')
    audio_model = models.MM_DeiT_ImageNet(label_dim=args.n_class, audio_length=args.target_length, stride=16, depth=12,
                                          fusion_depth=0,
                                          num_video_frames=10, pretrained=args.pretrain_path, mode=args.ftmode)
elif args.model == 'MM-ViT-Fusion':
    print('finetune MM-ViT-Fusion 12 modality-specific layers')
    audio_model = models.MM_ViT_Fusion_ImageNet(label_dim=args.n_class, audio_length=args.target_length, depth=12,
                                                fusion_depth=1, num_video_frames=10, pretrained=args.pretrain_path,
                                                mode=args.ftmode)
elif args.model == 'ViT-Kinetics':
    print('finetune ViT-Kinetics 12 modality-specific layers')
    audio_model = models.MM_ViT_Kinetics_ImageNet(label_dim=args.n_class,
                                                  depth=12,
                                                  fusion_depth=0,
                                                  num_video_frames=10,
                                                  pretrained=args.pretrain_path,
                                                  mode=args.ftmode)

elif args.model == 'MM-CLIP':
    print('finetune CLIP 24 modality-specific layers')
    audio_model = models.MM_CLIP(label_dim=args.n_class,
                                 layers=24,
                                 num_video_frames=10,
                                 embed_dim=1024,
                                 patch_size=14,
                                 heads=16,
                                 pretrained=args.pretrain_path,
                                 ftmode=args.ftmode)

elif args.model == 'MM-CLIP-AVE-Base':
    print('finetune CLIP 12 modality-specific layers')
    audio_model = models.MM_CLIP_AVE(label_dim=args.n_class,
                                     layers=12,
                                     num_video_frames=10,
                                     embed_dim=768,
                                     patch_size=16,
                                     heads=8,
                                     pretrained=args.pretrain_path,
                                     ftmode=args.ftmode)

elif args.model == 'MM-CLIP-AVE-Large':
    print('finetune CLIP 24 modality-specific layers')
    audio_model = models.MM_CLIP_AVE(label_dim=args.n_class,
                                     layers=24,
                                     num_video_frames=10,
                                     embed_dim=1024,
                                     patch_size=14,
                                     heads=16,
                                     pretrained=args.pretrain_path,
                                     ftmode=args.ftmode)

# ./logs/ave29_fusion_CLIP_adapt_large_Sep6.log: 81.8(average), 39M, mlp_ratio=0.125
# ./logs/ave29_fusion_CLIP_adapt_large_Sep7_light.log: 83.2, mlp_ratio=0.0625, 20.1M
# ./logs/ave29_fusion_CLIP_adapt_large_Sep7_lightv2.log:82.2 , mlp_ratio=0.03125, 10.7M

elif args.model == 'MM-Swin-AVE-Base':

    print('finetune Swin Base modality-specific layers')
    audio_model = models.SwinTransformer2D_Adapter_New(label_dim=args.n_class,
                                                       patch_size=[1, 4, 4],
                                                       num_frames=10,
                                                       embed_dim=128,
                                                       depths=[2, 2, 18, 2],
                                                       num_heads=[4, 8, 16, 32],
                                                       window_size=7,
                                                       pretrained=args.pretrain_path,
                                                       ftmode=args.ftmode,
                                                       adapter_mlp_ratio=[0.125, 0.125, 0.0625, 0.0625])
    # [0.25, 0.25, 0.125, 0.125] Sep 6 Vanilla (large?)
    # [0.5, 0.25, 0.125, 0.0625] Sep 7 Base (Light)
    # [0.125, 0.125, 0.0625, 0.0625] Sep 7 Lightv2

elif args.model == 'MM-Swin-AVE-Large':

    print('finetune Swin Large modality-specific layers')

    audio_model = models.SwinTransformer2D_Adapter_New(label_dim=args.n_class,
                                                       patch_size=[1, 4, 4],
                                                       img_size=224,
                                                       num_frames=10,
                                                       embed_dim=192,
                                                       depths=[2, 2, 18, 2],
                                                       num_heads=[6, 12, 24, 48],
                                                       window_size=7,
                                                       pretrained=args.pretrain_path,
                                                       ftmode=args.ftmode,
                                                       adapter_mlp_ratio=[0.5, 0.25, 0.125, 0.0625])
    # [0.25, 0.25, 0.25, 0.25] Aug 17
    # [0.25, 0.25, 0.125, 0.125] Aug 17 new, Sep5: 81.9
    # [0.5, 0.25, 0.125, 0.0625] Aug 17 new
    # [1/6, 1/6, 1/12, 1/12] Sep6 light 81.2
    # [0.5, 0.25, 0.125, 0.0625] Sep6 light2: 82.5, base
    # [0.125, 0.125, 0.0625, 0.0625] Sep6 light3: 82.0, tiny

    '''
    audio_model = models.SwinTransformerV2_Adapter(label_dim=args.n_class,
                                                   img_size=192,
                                                   patch_size=[1, 4, 4],
                                                   num_frames=10,
                                                   embed_dim=192,
                                                   depths=[2, 2, 18, 2],
                                                   num_heads=[6, 12, 24, 48],
                                                   window_size=12,
                                                   pretrained=args.pretrain_path,
                                                   ftmode=args.ftmode)'''

elif args.model == 'MM-Swin-AVS-Base':

    print('finetune Swin Base modality-specific layers')

    audio_model = AVSModelBase.SwinTransformer2D_Adapter_AVS_Base(
        patch_size=[1, 4, 4],
        img_size=224,
        num_frames=5,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        pretrained=args.pretrain_path,
        ftmode=args.ftmode,
        adapter_mlp_ratio=[0.25, 0.25, 0.125, 0.125],
        channel=256,
        opt=None, config=None, vis_dim=[64, 128, 320, 512],
        tpavi_stages=[0, 1, 2, 3],
        tpavi_vv_flag=False,
        tpavi_va_flag=True
    )

elif args.model == 'MM-Swin-AVS-Large':

    print('finetune Swin Large modality-specific layers')

    audio_model = AVSModelWithoutAdapt.SwinTransformer2D_Adapter_AVS_Without_Adapt(
        patch_size=[1, 4, 4],
        img_size=224,
        num_frames=5,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        pretrained=args.pretrain_path,
        ftmode=args.ftmode,
        adapter_mlp_ratio=[0.5, 0.25, 0.125, 0.0625],
        channel=256,
        opt=None, config=None, vis_dim=[64, 128, 320, 512],
        tpavi_stages=[0, 1, 2, 3],
        tpavi_vv_flag=False,
        tpavi_va_flag=True
    )

elif args.model == 'ViT-Kinetics-Text':
    print('finetune ViT-Kinetics 12 modality-specific layers with language aware')
    dataset_classes = train_loader.dataset.get_all_classname()
    '''
    imageb_model = imagebind_model.imagebind_huge(pretrained=True)
    imageb_model.eval()
    imageb_model.to('cuda:0')'''
    clip_model, preprocess = clip.load("ViT-L/14")

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset_classes):
            classname = [classname]
            text_input = clip.tokenize(classname).cuda()
            class_embeddings = clip_model.encode_text(text_input)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
    print('Extracted zero-shot weights for full classes')
    print(zeroshot_weights.shape)

    audio_model = models.MM_ViT_Kinetics_ImageNet(label_dim=args.n_class,
                                                  depth=12,
                                                  fusion_depth=1,
                                                  num_video_frames=10,
                                                  pretrained=args.pretrain_path,
                                                  mode=args.ftmode,
                                                  text_embed=zeroshot_weights)
else:
    raise ValueError('model not supported')

'''
# finetune based on a CAV-MAE pretrained model, which is the default setting unless for ablation study
if args.pretrain_path != 'None':
    # TODO: change this to a wget link
    mdl_weight = torch.load(args.pretrain_path)
    # TODO: filter MLP head for k400
    mlp_head_list = []
    for param in mdl_weight:
        if 'mlp_head' in param:
            mlp_head_list.append(param)

    for mlp_head in mlp_head_list:
        removed_key = mdl_weight.pop(mlp_head)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print('now load cav-mae pretrained weights from ', args.pretrain_path)
    print(miss, unexpected)
'''
if args.pretrain_path is None and args.finetune_path != 'None':
    # TODO: change this to a wget link
    mdl_weight = torch.load(args.finetune_path)
    mdl_weight['module.ln_post.weight'] = mdl_weight['module.norm.weight']
    mdl_weight['module.ln_post.bias'] = mdl_weight['module.norm.bias']
    # TODO: filter MLP head for k400
    mlp_head_list = []
    for param in mdl_weight:
        if 'mlp_head' in param:
            mlp_head_list.append(param)

    for mlp_head in mlp_head_list:
        removed_key = mdl_weight.pop(mlp_head)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print('now load k400 ViT pretrained weights from ', args.finetune_path)
    print(miss, unexpected)

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)
with open(args.exp_dir + '/args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

print('Now starting training for {:d} epochs.'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)  # start training

# todo start validation
# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
'''
def wa_model(exp_dir, start_epoch, end_epoch):
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location='cpu')
    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location='cpu')
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, start_epoch, end_epoch))
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    return sdA

# evaluate with multiple frames
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)
if args.wa == True:
    sdA = wa_model(args.exp_dir, start_epoch=args.wa_start, end_epoch=args.wa_end)
    torch.save(sdA, args.exp_dir + "/models/audio_model_wa.pth")
else:
    # if no wa, use the best checkpint
    sdA = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location='cpu')
msg = audio_model.load_state_dict(sdA, strict=True)
print(msg)
audio_model.eval()

# skil multi-frame evaluation, for audio-only model
if args.skip_frame_agg == True:
    #val_audio_conf['frame_use'] = 5
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
    if args.metrics == 'mAP':
        cur_res = np.mean([stat['AP'] for stat in stats])
        print('mAP is {:.4f}'.format(cur_res))
    elif args.metrics == 'acc':
        cur_res = stats[0]['acc']
        print('acc is {:.4f}'.format(cur_res))
else:
    res = []
    multiframe_pred = []
    total_frames = 1 # change if your total frame is different
    for frame in range(total_frames):
        #val_audio_conf['frame_use'] = frame
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
        print(audio_output.shape)
        if args.metrics == 'acc':
            audio_output = torch.nn.functional.softmax(audio_output.float(), dim=-1)
        elif args.metrics == 'mAP':
            audio_output = torch.nn.functional.sigmoid(audio_output.float())

        audio_output, target = audio_output.numpy(), target.numpy()
        multiframe_pred.append(audio_output)
        if args.metrics == 'mAP':
            cur_res = np.mean([stat['AP'] for stat in stats])
            print('mAP of frame {:d} is {:.4f}'.format(frame, cur_res))
        elif args.metrics == 'acc':
            cur_res = stats[0]['acc']
            print('acc of frame {:d} is {:.4f}'.format(frame, cur_res))
        res.append(cur_res)

    # ensemble over frames
    multiframe_pred = np.mean(multiframe_pred, axis=0)
    if args.metrics == 'acc':
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(multiframe_pred, 1))
        print('multi-frame acc is {:f}'.format(acc))
        res.append(acc)
    elif args.metrics == 'mAP':
        AP = []
        for k in range(args.n_class):
            # Average precision
            avg_precision = metrics.average_precision_score(target[:, k], multiframe_pred[:, k], average=None)
            AP.append(avg_precision)
        mAP = np.mean(AP)
        print('multi-frame mAP is {:.4f}'.format(mAP))
        res.append(mAP)
    np.savetxt(args.exp_dir + '/mul_frame_res.csv', res, delimiter=',')'''