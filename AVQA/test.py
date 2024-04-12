import torch
import json
import torch.nn as nn
import ast
import dataloader as dataloader
from dataloader import ToTensor
from torchvision import transforms
import model.Swin_AVQAModel as AVQAModel
import argparse
from torch.cuda.amp import autocast,GradScaler

def test(model, val_loader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if not isinstance(model, nn.DataParallel):
		model = nn.DataParallel(model)
	model = model.to(device)
	model.eval()
	total = 0
	correct = 0
	samples = json.load(open('./STG-CMA/AVQA/data/json/avqa-test.json', 'r'))
	A_count = []
	A_cmp = []
	V_count = []
	V_loc = []
	AV_ext = []
	AV_count = []
	AV_loc = []
	AV_cmp = []
	AV_temp = []
	with torch.no_grad():
		for batch_idx, sample in enumerate(val_loader):
			audio,visual_posi, visual_nega, target, question = sample['audio'].to(device), sample['visual_posi'].to(device), sample['visual_nega'].to(device), sample['label'].to(device), sample['question'].to(device)
			with autocast():
				preds_qa, out_match_posi, out_match_nega = model(audio, visual_posi,visual_nega, question, 'fusion')
			preds = preds_qa
			_, predicted = torch.max(preds.data, 1)

			total += preds.size(0)
			correct += (predicted == target).sum().item()

			x = samples[batch_idx]
			type =ast.literal_eval(x['type'])
			if type[0] == 'Audio':
				if type[1] == 'Counting':
					A_count.append((predicted == target).sum().item())
				elif type[1] == 'Comparative':
					A_cmp.append((predicted == target).sum().item())
			elif type[0] == 'Visual':
				if type[1] == 'Counting':
					V_count.append((predicted == target).sum().item())
				elif type[1] == 'Location':
					V_loc.append((predicted == target).sum().item())
			elif type[0] == 'Audio-Visual':
				if type[1] == 'Existential':
					AV_ext.append((predicted == target).sum().item())
				elif type[1] == 'Counting':
					AV_count.append((predicted == target).sum().item())
				elif type[1] == 'Location':
					AV_loc.append((predicted == target).sum().item())
				elif type[1] == 'Comparative':
					AV_cmp.append((predicted == target).sum().item())
				elif type[1] == 'Temporal':
					AV_temp.append((predicted == target).sum().item())

	print('Audio Counting Accuracy: %.2f %%' % (
			100 * sum(A_count)/len(A_count)))
	print('Audio Cmp Accuracy: %.2f %%' % (
			100 * sum(A_cmp) / len(A_cmp)))
	print('Audio Accuracy: %.2f %%' % (
			100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
	print('Visual Counting Accuracy: %.2f %%' % (
			100 * sum(V_count) / len(V_count)))
	print('Visual Loc Accuracy: %.2f %%' % (
			100 * sum(V_loc) / len(V_loc)))
	print('Visual Accuracy: %.2f %%' % (
			100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
	print('AV Ext Accuracy: %.2f %%' % (
			100 * sum(AV_ext) / len(AV_ext)))
	print('AV counting Accuracy: %.2f %%' % (
			100 * sum(AV_count) / len(AV_count)))
	print('AV Loc Accuracy: %.2f %%' % (
			100 * sum(AV_loc) / len(AV_loc)))
	print('AV Cmp Accuracy: %.2f %%' % (
			100 * sum(AV_cmp) / len(AV_cmp)))
	print('AV Temporal Accuracy: %.2f %%' % (
			100 * sum(AV_temp) / len(AV_temp)))

	print('AV Accuracy: %.2f %%' % (
			100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
				   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

	print('Overall Accuracy: %.2f %%' % (
			100 * correct / total))

	return 100 * correct / total

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--dataset_mean", type=float, default=-5.214385986328125, help="the dataset mean, used for input normalization in audio branch")
	parser.add_argument("--dataset_std", type=float, default=3.8699076175689697, help="the dataset std, used for input normalization in audio branch")
	args = parser.parse_args()

	data_val = './STG-CMA/AVQA/data/json/avqa-test.json'
	dir_image = './STG-CMA/AVQA/dataset'
	dir_audio_wav = './STG-CMA/AVQA/AVQA_Dataset'
	grounding_pretrained = '/AVQA/grounding_gen/models_grounding_gen/main_grounding_gen_best.pt'
	pretrain_path = './STG-CMA/pretrained_model/Swin_ViT/swin_large_patch4_window7_224_22k.pth'

	test_dataset = dataloader.AVQA_dataset(label=data_val, video_res14x14_dir=dir_image, audio_wave_dir=dir_audio_wav,
								args=args, transform=transforms.Compose([ToTensor()]), mode_flag='test')
	print(test_dataset.__len__())
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

	model = AVQAModel.SwinTransformer2D_Adapter_AVQA(
                                                   patch_size=[1, 4, 4],
                                                   img_size=224,
                                                   num_frames=10,
                                                   embed_dim=192,
                                                   depths=[2, 2, 18, 2],
                                                   num_heads=[6, 12, 24, 48],
                                                   window_size=7,
                                                   pretrained=pretrain_path,
                                                   grounding_pretrained=grounding_pretrained,
                                                   ftmode='fusion',
                                                   adapter_mlp_ratio=[0.5, 0.25, 0.125, 0.0625])

	best_checkpoint = './STG-CMA/AVQA/exp/ave_29_ft-bal-MM-Swin-AVQA-Large-1e-4-10-0.7-1-bs2-ldaFalse-fusion-fzTrue-h0.1-a5-Swin-Large-AVQA-DEC2/models/best_audio_model.pth'
	if not isinstance(model, torch.nn.DataParallel):
		model = torch.nn.DataParallel(model)
	model.load_state_dict(torch.load(best_checkpoint))
	test(model, test_loader)