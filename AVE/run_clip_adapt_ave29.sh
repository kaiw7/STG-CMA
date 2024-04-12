model=MM-CLIP-AVE-Large #MM-CLIP-AVE-Large or MM-CLIP-AVE-Base
ftmode=fusion #fusion, audioonly, videoonly, multimodal
pretrain_path=./STG-CMA/pretrained_model/CLIP # put downloaded CLIP model here

freeze_base=True #True is for adaptation
head_lr=0.1 # the scaled lr for MLP classifier

lr=5e-5
epoch=20

wa=True
wa_start=8
wa_end=20
lr_adapt=False

lr_cosine_adapt=True
min_lr=2e-6
warmup_epochs=2

dataset_mean=-4.1426 #the mean for AVE datasert
dataset_std=3.2001 #the std for AVE dataset
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=1 #2
label_smooth=0.1

dataset=ave-29
# put train/test data and label csv here
tr_data=./STG-CMA/AVE/preprocess/train_order.h5
te_data=./STG-CMA/AVE/preprocess/test_order.h5
label_csv=./STG-CMA/AVE/preprocess/labels.h5
log_name=CLIP-Large-AVE # for example

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

# todo head_lr=0.1, lr=5e-5, min_lr=2e-6, warmup_epochs=2, batch_size=1, epoch=20
#./logs/ave29_fusion_CLIP_adapt_Sep2_large_new.log: 82.2, average, 39M
# ./logs/ave29_fusion_CLIP_adapt_large_Sep6.log: 81.8(average), 39M, mlp_ratio=0.125
# acc: 83.3, mlp_ratio=0.0625, 20.1M, Base
# acc: 82.2 , mlp_ratio=0.03125, 10.7M, Tiny
# acc: 78.7 , 11.5M, mlp_ratio=0.125, Base
# acc: 76.3, 3.5M, mlp_ratio=0.03125, Tiny
