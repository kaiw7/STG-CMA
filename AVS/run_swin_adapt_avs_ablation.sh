model=MM-Swin-AVS-Large
ftmode=fusion #fusion, audioonly, videoonly, multimodal
pretrain_path=./STG-CMA/pretrained_model/Swin_ViT/swin_large_patch4_window7_224_22k.pth #swin_base_patch4_window7_224_22k.pth

freeze_base=True
head_lr=0.1
bal=None
lr=3e-4 #TODO base:1e-4; large:5e-5
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
batch_size=2 # TODO base: 3; large:2
label_smooth=0.1

dataset=avsbench
dir_image=./STG-CMA/AVS/dataset/Single-source/s4_data/visual_frames
dir_audio_log_mel=./STG-CMA/AVS/dataset/Single-source/s4_data/audio_log_mel
dir_audio_wav=./STG-CMA/AVS/dataset/Single-source/s4_data/audio_wav
dir_mask=./STG-CMA/AVS/dataset/Single-source/s4_data/gt_masks
log_name=Swin-Large-AVS # for example
exp_dir=./exp/ave_29_ft-bal-${model}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-log${log_name}

CUDA_CACHE_DISABLE=1 nohup python -u ./run_adapt_avs_ablation.py --model ${model} --dataset ${dataset} \
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

# todo AVS ablation study new
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec25_without_adapt.log: 21.2M, MIoU: 79.8; only frozen vit
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec25_add_temporal_with_adapt.log, 24.7M; MIoU: 80.8
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec27_spatialtemporal_adapt.log, 31.6M; MIoU: 81.4
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec27_temporalglobal_adapt.log, 31.6M; MIoU: 81.5
#


# todo AVS ablation study
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec25_without_adapt.log: MIoU: 79.8
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec25_add_temporal_without_adapt.log: 21.2M; MIoU: 55.9
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec25_add_temporal_with_adapt.log, 24.7M; MIoU: 80.8
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec26_spatialonly_adapt.log, 28.1M; MIoU: 76.5
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec26_globalonly_adapt.log, 28.1M; MIoU: 76.1
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec27_spatialtemporal_adapt.log, 31.6M; MIoU: 81.4
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec27_temporalglobal_adapt.log, 31.6M; MIoU: 81.5
# ./logs/ave29_fusion_Swin_adapt_large_avs_Dec28_spatialglobal_adapt.log, 35.1; MIoU: 77.0
#