model=MM-Swin-AVQA-Large #MM-Swin-AVS-Base
ftmode=fusion #fusion, audioonly, videoonly, multimodal
pretrain_path=./STG-CMA/pretrained_model/Swin_ViT/swin_large_patch4_window7_224_22k.pth #swin_base_patch4_window7_224_22k.pth

freeze_base=True
head_lr=0.1
bal=None
lr=2.5e-5
epoch=20


wa=False #True
wa_start=8
wa_end=20
lr_adapt=False

lr_cosine_adapt=True
min_lr=2e-6
warmup_epochs=2


dataset_mean=-5.214385986328125
dataset_std=3.8699076175689697

target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=2
label_smooth=0.1

dataset=music-avqa

dir_image=./STG-CMA/AVQA/dataset
dir_audio_wav=./STG-CMA/AVQA/AVQA_Dataset
data_train=./STG-CMA/AVQA/data/json/avqa-train.json
data_val=./STG-CMA/AVQA/data/json/avqa-test.json
grounding_pretrained=./STG-CMA/AVQA/grounding_gen/models_grounding_gen/lavish_grounding_gen_best_dgstc.pt #main_grounding_gen_best_lavish.pt #main_grounding_gen_best_lavish.pt or lavish_grounding_gen_best_dgstc.pt
log_name=Swin-Large-AVE # for example
exp_dir=./exp/ave_29_ft-bal-${model}-${lr}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-log${log_name}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 nohup python -u ./run_adapt_avqa.py --model ${model} --dataset ${dataset} \
--dir_image ${dir_image}  --exp-dir $exp_dir \
--dir_audio_wav ${dir_audio_wav} --grounding_pretrained ${grounding_pretrained} \
--data_train ${data_train} --data_val ${data_val} \
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
