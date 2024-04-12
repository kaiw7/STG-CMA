import sys
import os
import datetime
sys.path.append('./AudioVisual') # todo change into your path
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from einops import rearrange
from loss import IouSemanticAwareLoss, mask_iou

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, loss_meter, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    iou_loss_meter, sa_loss_meter = AverageMeter(), AverageMeter()

    progress = []
    best_epoch, best_mAP, best_acc, best_miou = 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    model_type = args.model

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_miou, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                'mlp_head.2.weight', 'mlp_head.2.bias', 'mlp_head.3.weight', 'mlp_head.3.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]

    #base_params = [i[1] for i in base_params]
    base_parameters = []
    adapt_parameters = []

    # todo:  and 'ln_post' not in name; and 'patch_embed_audio' not in name
    # todo: 'temporal_position_bias_table'
    # and 'temporal_position_bias_table' not in name
    for name, param in base_params:
        if 'adapter' not in name and 'temporal_embedding' not in name and 'ln_post' not in name and 'Adapter' not in name and 'my_tokens' not in name and 'gate_' not in name and 'ln_before' not in name and 'temporal_position_bias_table' not in name and 'avstask_' not in name:
            base_parameters.append(param)
        else:
            adapt_parameters.append(param)

    # if freeze the pretrained parameters and only train the newly initialized model (linear probing)
    if args.freeze_base == True:
        print('Pretrained backbone parameters are frozen.')
        for param in base_parameters:
            param.requires_grad = False

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(args.head_lr))
    optimizer = torch.optim.Adam([{'params': adapt_parameters, 'lr': args.lr}], weight_decay=5e-7, betas=(0.95, 0.999)) # todo ours weight_decay=5e-7; Nov 28:5e-5
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = base_lr # optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)

    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total newly initialized Adapter parameter number is : {:.3f} million'.format(sum(p.numel() for p in adapt_parameters) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_parameters) / 1e6))

    # only for preliminary test, formal exps should use fixed learning rate scheduler
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    if args.lr_adapt == False and args.lr_cosine_adapt == True:
        num_training_steps_per_epoch = len(train_loader.dataset) // args.batch_size
        lr_schedule_values = cosine_scheduler(
            base_value=args.lr,
            final_value=args.min_lr,
            epochs=args.n_epochs,
            niter_per_ep=num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs,
            start_warmup_value=0,
            warmup_steps=-1
        )
        lr_schedule_values_mlp_head = cosine_scheduler(
            base_value=args.lr * args.head_lr,
            final_value=args.min_lr,
            epochs=args.n_epochs,
            niter_per_ep=num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs,
            start_warmup_value=0,
            warmup_steps=-1
        )
        print('Cosine Learning rate scheduler is used.')
        scheduler = 'Cosine Learning Rate Scheduler'

    if args.lr_adapt == False and args.lr_cosine_adapt == False:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == 'IoU':
        loss_fn = IouSemanticAwareLoss
    args.loss_fn = loss_fn

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 2])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (imgs, audio_spec, audio, mask) in enumerate(train_loader):
            # imgs: [b, 5, 3, 224, 224]; audio_spec: [b, 5, 224, 224], mask: [b, 1, 1, 224, 224]
            # cosine learning rate scheduling
            if args.lr_adapt == False and args.lr_cosine_adapt == True:

                for idx, param_group in enumerate(optimizer.param_groups):
                    if idx == 0:
                        param_group["lr"] = lr_schedule_values[global_step]
                    #else:
                    #    param_group["lr"] = lr_schedule_values_mlp_head[global_step]

            imgs = imgs.to(device, non_blocking=True)
            audio_spec = audio_spec.to(device, non_blocking=True)
            #audio = audio.cuda()
            mask = mask.to(device, non_blocking=True) # [b, 1, 1, 224, 224]
            B, frame, C, H, W = imgs.shape # [b, 5, 3, 224, 224]
            mask = mask.view(B, H, W) # [b, 224, 224]

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / imgs.shape[0])
            dnn_start_time = time.time()

            #with autocast():
            output, visual_map_list, a_fea_list = audio_model(audio_spec, imgs, args.ftmode) # [bf, 1, 224, 224]
            loss, loss_dict = IouSemanticAwareLoss(output, mask.unsqueeze(1).unsqueeze(1),
                                                   a_fea_list, visual_map_list,
                                                   lambda_1=0,  # todo args
                                                   count_stages=[],
                                                   sa_loss_flag=False,
                                                   mask_pooling_type='avg')

            #optimizer.zero_grad()
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            iou_loss_meter.update(loss_dict['iou_loss'])
            sa_loss_meter.update(loss_dict['sa_loss'])

            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/imgs.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/imgs.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'
                  'IoU Loss {iou_loss_meter.val:.4f}\t'
                  'SA Loss {sa_loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter, iou_loss_meter=iou_loss_meter, sa_loss_meter=sa_loss_meter), flush=True)

                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')

        miou_score = validate(audio_model, test_loader, args)

        # todo
        print("miou_score: {:.6f}".format(miou_score))

        result[epoch-1, :] = [miou_score, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if miou_score > best_miou:
            best_miou = miou_score
            if main_metrics == 'miou':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))


        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        #print('Epoch-{0} head_lr: {1}'.format(epoch, optimizer.param_groups[1]['lr']))

        #with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
        #    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        iou_loss_meter.reset()
        sa_loss_meter.reset()
        per_sample_dnn_time.reset()

def validate(audio_model, val_loader, args, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    miou_meter = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    #A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (imgs, audio_spec, audio, mask, _, _) in enumerate(val_loader):
            imgs = imgs.to(device) # [b, 5, 3, 224, 224]
            audio_spec = audio_spec.to(device) # [b, 5, 224, 224]
            mask = mask.to(device) # [b, 5, 1, 224, 224]
            B, frame, C, H, W = imgs.shape
            mask = mask.view(B * frame, H, W) # [bf, 224, 224]

            #with autocast():
            output, _, _ = audio_model(audio_spec, imgs, args.ftmode) # [bf, 1, 224, 224]

            miou = mask_iou(output.squeeze(1), mask) # [bf, 224, 224] , mask: [bf, 224, 224]
            #print(miou)
            # todo
            miou_meter.update(miou.item())

            batch_time.update(time.time() - end)
            end = time.time()

        miou_score = miou_meter.avg

    if output_pred:
        return output, miou_score
    else:
        return miou_score