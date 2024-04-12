model=MM-Swin-AVS-Large #MM-Swin-AVS-Base or MM-Swin-AVS-Large
ftmode=fusion #fusion, audioonly, videoonly, multimodal
pretrain_path=./STG-CMA/pretrained_model/Swin_ViT/swin_large_patch4_window7_224_22k.pth #swin_base_patch4_window7_224_22k.pth
freeze_base=True
head_lr=0.1
bal=None
lr=3e-4
epoch=20

wa=False #True
wa_start=8
wa_end=20
lr_adapt=False

lr_cosine_adapt=True
min_lr=2e-5
warmup_epochs=5

dataset_mean=-5.669627666473389
dataset_std=3.948380470275879
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=2 #TODO base: 3; large:2
label_smooth=0.1

dataset=avsbench
dir_image=./STG-CMA/AVS/dataset/Single-source/s4_data/visual_frames
dir_audio_log_mel=./STG-CMA/AVS/dataset/Single-source/s4_data/audio_log_mel
dir_audio_wav=./STG-CMA/AVS/dataset/Single-source/s4_data/audio_wav
dir_mask=./STG-CMA/AVS/dataset/Single-source/s4_data/gt_masks
log_name=Swin-Large-AVS # for example
exp_dir=./exp/ave_29_ft-bal-${model}-${lr}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-log${log_name}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 nohup python -u ./run_adapt_avs.py --model ${model} --dataset ${dataset} \
--dir_image ${dir_image} --dir_audio_log_mel ${dir_audio_log_mel} --exp-dir $exp_dir \
--dir_audio_wav ${dir_audio_wav} --dir_mask ${dir_mask} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss IoU --metrics miou --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --finetune_path ${finetune_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} --lr_cosine_adapt ${lr_cosine_adapt} --min_lr ${min_lr} --warmup_epochs ${warmup_epochs} \
--num-workers 16 > ./logs/${log_name}.log 2>&1 &

# todo final version
#  lr=3e-4, min_lr=2e-5, warmup_epochs=5, batch_size=2, MIoU: 81.8; Param: 38.6M
#  lr=3e-4, min_lr=2e-5, warmup_epochs=5, batch_size=2, MIoU: 81.0; Param: 29.7M



