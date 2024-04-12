import os
import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import torchaudio
import soundfile as sf

import cv2
from PIL import Image
from torchvision import transforms

#from config import cfg
from ipdb import set_trace
from einops import rearrange
import warnings
import pdb

warnings.filterwarnings('ignore')


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
    return audio_log_mel


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""

    def __init__(self, audio_conf):
        super(S4Dataset, self).__init__()
        self.audio_conf = audio_conf

        self.split = self.audio_conf.get('mode') # todo train or val
        print('now in {:s} mode.'.format(self.split))

        self.mask_num = 1 if self.split == 'train' else 5 # todo

        df_all = pd.read_csv('./STG-CMA/AVS/dataset/Single-source/s4_meta_data.csv', # todo put your downloaded file here
                             sep=',') # todo 's4_meta_data.csv'

        self.df_split = df_all[df_all['split'] == self.split]

        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))

        self.dir_image = self.audio_conf.get('dir_image')
        self.dir_audio_log_mel = self.audio_conf.get('dir_audio_log_mel')
        self.dir_audio_wav = self.audio_conf.get('dir_audio_wav')
        self.dir_mask = self.audio_conf.get('dir_mask')

        # todo changed into video version
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # todo add
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.audio_length = 1.95 # todo
        #self.opt = args

        # ### ---> yb calculate: AVE dataset for 192
        # self.norm_mean =  -4.984795570373535
        # self.norm_std =  3.7079780101776123
        # ### <----

        ### ---> yb calculate: AVE dataset for 192 TODO comment for sta calculation
        self.norm_mean = self.audio_conf.get('mean') # -5.669627666473389
        self.norm_std = self.audio_conf.get('std') # 3.948380470275879

    ### <----

    def _wav2fbank(self, filename, filename2=None, idx=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            # mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        ## yb: align ##
        # if waveform.shape[1] > sr*(self.opt.audio_length+0.1):
        sample_indx = np.linspace(0, waveform.shape[1] - sr * (self.audio_length + 0.1), num=5, dtype=int)
        waveform = waveform[:, sample_indx[idx]:sample_indx[idx] + int(sr * self.audio_length)]
        #print(waveform.shape)
        # waveform = waveform.mean(dim=0, keepdim=True)
        ## align end ##

        # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=224, dither=0.0, frame_shift=4.4) # todo: 224, 4.4 or 10

        #print(fbank.shape)
        ########### ------> very important: audio normalized TODO comment for sta calculation
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ### <--------

        # target_length = int(1024 * (self.opt.audio_length/10)) ## for audioset: 10s
        target_length = 224 # 192  ## yb: overwrite for swin

        # target_length = 512 ## 5s
        # target_length = 256 ## 2.5s
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path = os.path.join(self.dir_image, self.split, category, video_name)
        audio_lm_path = os.path.join(self.dir_audio_log_mel, self.split, category, video_name + '.pkl')
        mask_base_path = os.path.join(self.dir_mask, self.split, category, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png" % (video_name, img_id)),
                                              transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png" % (video_name, mask_id)),
                                               transform=self.mask_transform, mode='1')
            masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0) # todo
        masks_tensor = torch.stack(masks, dim=0) # todo

        ### ---> loading all audio frames
        total_audio = []
        for audio_sec in range(5):
            fbank, mix_lambda = self._wav2fbank(
                os.path.join(self.dir_audio_wav, self.split, category, video_name + '.wav'), idx=audio_sec)
            total_audio.append(fbank)
        total_audio = torch.stack(total_audio)
        ### <----

        if self.split == 'train':
            return imgs_tensor, total_audio, audio_log_mel, masks_tensor
        else:
            return imgs_tensor, total_audio, audio_log_mel, masks_tensor, category, video_name

    def __len__(self):
        return len(self.df_split)


if __name__ == "__main__":
    dir_image = './STG-CMA/AVS/dataset/Single-source/s4_data/visual_frames'
    dir_audio_log_mel = './STG-CMA/AVS/dataset/Single-source/s4_data/audio_log_mel'
    dir_audio_wav = './STG-CMA/AVS/dataset/Single-source/s4_data/audio_wav'
    dir_mask = './STG-CMA/AVS/dataset/Single-source/s4_data/gt_masks'
    mean = -5.669627666473389 # todo, 224: -5.669627666473389 (frame_shift=4.4); -4.88803100585937 5 (frame_shift=10)
    std = 3.948380470275879 # todo  224: 3.948380470275879 (frame_shift=4.4); 4.292664051055908 (frame_shift=10)
    audio_conf = {'mode': 'train', 'dir_image': dir_image, 'dir_audio_log_mel': dir_audio_log_mel,
                  'dir_audio_wav': dir_audio_wav, 'dir_mask': dir_mask, 'mean': mean, 'std': std}
    train_dataset = S4Dataset(audio_conf)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=10,
                                                   shuffle=False,
                                                   num_workers=8,
                                                   pin_memory=True)
    mean = []
    std = []

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio_spec, audio_log_mel, mask = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        '''print(imgs.shape)
        print(audio_log_mel.shape)
        print(audio.shape)
        print(mask.shape)
        #print(mask)
        #print(category)
        #print(video_name)
        break
        #pdb.set_trace()
        '''
        b, t, w, h = audio_spec.shape

        audio_spec = rearrange(audio_spec, 'b t w h -> (b t) (w h)')

        cur_mean = torch.mean(audio_spec, dim=-1)
        cur_std = torch.std(audio_spec, dim=-1)
        mean.append(cur_mean)
        std.append(cur_std)
    print('mean: ', torch.hstack(mean).mean().item(), 'std: ', torch.hstack(std).mean().item())
    print('n_iter', n_iter)