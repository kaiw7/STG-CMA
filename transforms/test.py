import numpy as np
import torch
import timm
import video_transforms
from torchvision import transforms
import volume_transforms
import numpy as nn

aug_transform = video_transforms.create_random_augment(
            input_size=(224, 224),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bilinear',
        )
print(aug_transform)

image_tensor = torch.randn((3, 224, 224))
image_tensor = transforms.ToPILImage()(image_tensor)
out = np.array(image_tensor)
print(out.shape)

'''
aug_image = aug_transform([image_tensor])
print(aug_image)
aug_transform = video_transforms.Compose([
                video_transforms.Resize(224, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(224, 224)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
image_list = nn.array(torch.randn((8, 224, 224, 3)))
out = aug_transform(image_list)
print(out.shape)'''
