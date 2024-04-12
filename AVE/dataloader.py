import csv
import json
import os.path
import pandas as pd
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL
from transforms import video_transforms, random_erasing
from torchvision import transforms
from transforms import volume_transforms
import glob
import h5py


def read_label_video_id_from_txt_file(input_file_list_path):
    label_video_id_list = []
    with open(input_file_list_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            label_video_id_list.append(line)

    return label_video_id_list

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_index_dict_modified(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',') # [class_name, label_index]
            index_lookup[line[0]] = line[1]
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_h5py_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_h5py_file

        with h5py.File(dataset_h5py_file, 'r') as hf:
            self.data = hf['order'][:]

        print('Dataset has {:d} samples'.format(len(self.data)))

        self.num_samples = len(self.data)

        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)

        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        # change: todo
        with h5py.File(label_csv, 'r') as hf:
            self.index_dict = hf['avadataset'][:] # one-hot label

        self.label_num = self.index_dict.shape[-1]
        print('number of classes is {:d}'.format(self.label_num))

        # raw ground truth
        # todo change into your path
        self.raw_gt = pd.read_csv("./STG-CMA/AVE/preprocess/Annotations.txt", sep="&", header=None)


        self.target_length = self.audio_conf.get('target_length')
        self.audio_length = 1

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # multimodal, or videoonly, or audioonly
        self.ftmode = self.audio_conf.get('ftmode')
        print('now in {:s} finetuning mode.'.format(self.ftmode))

        self.model_type = self.audio_conf.get('model_tpye')

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get('frame_use', -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get('total_frame', 10)
        print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))

        if self.mode == 'train':
            # numpy
            self.preprocess = self._aug_frame_train
        elif self.mode == 'eval':
            # numpy
            self.preprocess = video_transforms.Compose([
                video_transforms.Resize(224, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(224, 224)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # return the list of class names
    def get_all_classname(self):
        classname_list = []
        for keys in self.index_dict.keys():
            classname_list.append(keys)
        return classname_list

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def _wav2fbank(self, filename, filename2=None, idx=None):
        # no mixup
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

            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        if waveform.shape[1] > 16000 * (self.audio_length + 0.1):
            sample_indx = np.linspace(0, waveform.shape[1] - 16000 * (self.audio_length + 0.1), num=10, dtype=int)
            waveform = waveform[:, sample_indx[idx]:sample_indx[idx] + int(16000 * self.audio_length)]

        try:
            # for pre-trained swin backbone
            if self.model_type == 'MM-Swin-AVE-Base' or self.model_type == 'MM-Swin-AVE-Large':
                fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                                          use_energy=False,
                                                          window_type='hanning', num_mel_bins=224, dither=0.0,
                                                          frame_shift=4.4) #TODO 4.4
            else:
                # for pre-trained clip backbone
                fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                      window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([102, 128]) + 0.01
            print('there is a loading error')

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        #fbank = (fbank - self.norm_mean) / (self.norm_std)

        if self.model_type == 'MM-Swin-AVE-Base' or self.model_type == 'MM-Swin-AVE-Large':
            target_length = 224
        else:
            target_length = int(self.target_length * (1/10))

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

    def randselect_img(self, video_id, video_path):
        if self.mode == 'eval':
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, 9)

        while os.path.exists(video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg') == False and frame_idx >= 1:
            print('frame {:s} {:d} does not exist'.format(video_id, frame_idx))
            frame_idx -= 1
        out_path = video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg'
        #print(out_path)
        return out_path

# todo video_path: put the data under the dataset folder
    def select_all_img_frames(self, video_id, video_path='./STG-CMA/AVE/dataset/LAVISHData/AVE_Dataset/video_frames'):
        total_num_frames = len(glob.glob(video_path + '/' + video_id + '/*.jpg'))
        sample_indx = np.linspace(1, total_num_frames, num=10, dtype=int)

        all_frames_path_list = []
        for frame_idx in range(10):
            tmp_idx = sample_indx[frame_idx]
            frame_path = video_path + '/' + video_id + '/' + str("{:04d}".format(tmp_idx)) + '.jpg'
            all_frames_path_list.append(frame_path)
        #print(all_frames_path_list)
        return all_frames_path_list

    def get_all_image_frames(self, filename_list, filename2_list=None, mix_lambda=1):
        # filename: [frame1, frame2, ......]; filename2: [frame1, frame2, ......]
        image_tensor_list = []
        image_tensor_list1 = []
        image_tensor_list2 = []
        if filename2_list == None:
            for filename in filename_list:
                try:
                    img = Image.open(filename)
                    image_tensor = np.array(img)
                except:
                    image_tensor = np.array(torch.zeros([3, self.im_res, self.im_res]) + 0.01)
                    print('there is an error in loading image frames')

                image_tensor_list.append(image_tensor)
            video_tensor = self.preprocess(image_tensor_list)

            return video_tensor
        else:
            for idx, filename in enumerate(filename_list):
                try:
                    img1 = Image.open(filename)
                    image_tensor1 = np.array(img1)
                except:
                    image_tensor1 = np.array(torch.zeros([3, self.im_res, self.im_res]) + 0.01)
                    print('there is an error in loading image frames')

                try:
                    img2 = Image.open(filename2_list[idx])
                    image_tensor2 = np.array(img2)
                except:
                    image_tensor2 = np.array(torch.zeros([3, self.im_res, self.im_res]) + 0.01)
                    print('there is an error in loading image frames')

                #image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
                image_tensor_list1.append(image_tensor1)
                image_tensor_list2.append(image_tensor2)
            video_tensor1 = self.preprocess(image_tensor_list1)
            video_tensor2 = self.preprocess(image_tensor_list2)
            video_tensor = mix_lambda * video_tensor1 + (1 - mix_lambda) * video_tensor2
            return video_tensor

    def _aug_frame_train(self, buffer):
        # buffer # T C H W
        aug_transform = video_transforms.create_random_augment(
            input_size=(224, 224),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = self.tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)

        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )
        buffer = self.spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        erase_transform = random_erasing.RandomErasing(
            0.25,
            mode='pixel',
            max_count=1,
            num_splits=1,
            device="cpu",
        )
        buffer = buffer.permute(1, 0, 2, 3)
        buffer = erase_transform(buffer)
        buffer = buffer.permute(1, 0, 2, 3)
        return buffer

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=None,
            scale=None,
            motion_shift=False,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale,
                max_scale].
            aspect_ratio (list): Aspect ratio range for resizing.
            scale (list): Scale range for resizing.
            motion_shift (bool): Whether to apply motion shift for resizing.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            if aspect_ratio is None and scale is None:
                frames, _ = video_transforms.random_short_side_scale_jitter(
                    images=frames,
                    min_size=min_scale,
                    max_size=max_scale,
                    inverse_uniform_sampling=inverse_uniform_sampling,
                )
                frames, _ = video_transforms.random_crop(frames, crop_size)
            else:
                transform_func = (
                    video_transforms.random_resized_crop_with_shift
                    if motion_shift else video_transforms.random_resized_crop)
                frames = transform_func(
                    images=frames,
                    target_height=crop_size,
                    target_width=crop_size,
                    scale=scale,
                    ratio=aspect_ratio,
                )
            if random_horizontal_flip:
                frames, _ = video_transforms.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = video_transforms.random_short_side_scale_jitter(
                frames, min_scale, max_scale)
            frames, _ = video_transforms.uniform_crop(frames, crop_size,
                                                      spatial_idx)
        return frames

    def tensor_normalize(self, tensor, mean, std):
        """
        Normalize a given tensor by subtracting the mean and dividing the std.
        Args:
            tensor (tensor): tensor to normalize.
            mean (tensor or list): mean value to subtract.
            std (tensor or list): std to divide.
        """
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
            tensor = tensor / 255.0
        if type(mean) == list:
            mean = torch.tensor(mean)
        if type(std) == list:
            std = torch.tensor(std)
        tensor = tensor - mean
        tensor = tensor / std
        return tensor

    def __getitem__(self, index):
        real_idx = self.data[index]
        file_name = self.raw_gt.iloc[real_idx][1]

        if random.random() < self.mixup:
            mix_sample_idx = random.randint(0, self.num_samples - 1)
            mix_sample_idx = self.data[mix_sample_idx]
            mix_file_name = self.raw_gt.iloc[mix_sample_idx][1]
            # todo put audio data under the dataset folder
            mix_file_name = './STG-CMA/AVE/dataset/LAVISHData/AVE_Dataset/raw_audio' + '/' + mix_file_name + '.wav'
        else:
            mix_file_name = None
        # visual data
        if self.ftmode == 'multimodal' or self.ftmode == 'videoonly' or self.ftmode == 'fusion':
            try:
                image = self.get_all_image_frames(self.select_all_img_frames(video_id=file_name), None, 0)
            except:
                image = torch.zeros([3, 10, self.im_res, self.im_res]) + 0.01
                print('there is an error in loading image')
        else:
            image = torch.zeros([3, 10, self.im_res, self.im_res]) + 0.01

        # audio data
        if self.ftmode == 'multimodal' or self.ftmode == 'audioonly' or self.ftmode == 'fusion':
            total_audio = []
            for audio_sec in range(10):
                fbank, mix_lambda = self._wav2fbank('./STG-CMA/AVE/dataset/LAVISHData/AVE_Dataset/raw_audio' + '/' + file_name + '.wav', filename2=mix_file_name, idx=audio_sec)
                total_audio.append(fbank)
            total_audio = torch.stack(total_audio) # [10, length, dim]

        else:
            total_audio = torch.zeros([self.target_length, 128]) + 0.01

        label_indices = torch.from_numpy(self.index_dict[real_idx])

        return total_audio, image, label_indices

    def __len__(self):
        return self.num_samples