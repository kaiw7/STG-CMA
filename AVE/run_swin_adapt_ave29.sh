model=MM-Swin-AVE-Large #MM-Swin-AVE-Large or MM-Swin-AVE-Base
ftmode=multimodal #fusion, audioonly, videoonly, multimodal
pretrain_path=./STG-CMA/pretrained_model/Swin_ViT/swin_large_patch4_window7_224_22k.pth #or swin_base_patch4_window7_224_22k.pth

freeze_base=True
head_lr=0.1
lr=5e-5 #base:1e-4; large:5e-5
epoch=20

wa=True
wa_start=8
wa_end=20
lr_adapt=False

lr_cosine_adapt=True
min_lr=2e-6
warmup_epochs=2

dataset_mean=-4.1426
dataset_std=3.2001
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=1 # TODO base: 3; large:2
label_smooth=0.1

dataset=ave-29
tr_data=./STG-CMA/AVE/preprocess/train_order.h5
te_data=./MOSTG-CMAMA/AVE/preprocess/test_order.h5
label_csv=./STG-CMA/AVE/preprocess/labels.h5
log_name=Swin-Large-AVE # for example
exp_dir=./exp/ave_29_ft-bal-${model}-${lr}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-log${log_name}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 nohup python -u ./run_adapt_ave29.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 29 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss CE --metrics acc --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --finetune_path ${finetune_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} --lr_cosine_adapt ${lr_cosine_adapt} --min_lr ${min_lr} --warmup_epochs ${warmup_epochs} \
--num-workers 16 > ./logs/${log_name}.log 2>&1 &

# todo head_lr=0.1, lr=5e-5, min_lr=2e-6, warmup_epochs=2, batch_size=1, mixup=0.5
# acc: 82.5, 19M; [0.5, 0.25, 0.125, 0.0625] Swin-Large # todo Base adapter setting
# acc: 82.0 , 11.74M, [0.125, 0.125, 0.0625, 0.0625] Swin-Tiny # todo Tiny adapter setting
# acc: 81.4, 10.07M, [0.25, 0.25, 0.125, 0.125] Swin-Base # todo Base adapter setting
# acc: 81.1, 5.6M, [0.125, 0.125, 0.0625, 0.0625] Swin-Tiny # todo Tiny adapter setting
