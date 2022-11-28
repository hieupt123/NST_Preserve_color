import os.path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from VGG import VGG
import cal_mean_std
import skimage
import cv2
import torchvision.transforms as T
import numpy as np

# model = models.vgg19(pretrained=True).features
model = models.vgg19(pretrained=True)
# print(model)
# sys.exit()
# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG, self).__init__()
#
#         self.chosen_features = ['0', '5', '10', '19', '28']
#         self.model = models.vgg19(pretrained=True).features[:29]
#
#     def forward(self, x):
#         features = []
#
#         for layer_num, layer in enumerate(self.model):
#             x = layer(x)
#
#             if str(layer_num) in self.chosen_features:
#                 features.append(x)
#         return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def getFileName(path):
    name = path.split("/")[-1]
    file_name = os.path.splitext(name)[0]
    return file_name

device = torch.device("cuda" if torch.cuda.is_available else 'cpu')
image_size = 256

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[], std=[])
    ]
)

path_img = 'content-images/lena.jpg'
path_style = 'style-images/mosaic.jpg'


img = Image.open(path_style).convert('RGB')
original_img = cal_mean_std.conver_Yuv(path_img)
style_img_tranform = cal_mean_std.luminance_channel(path_img, path_style)


# convert from nparray to PIL
style_img_PIL = Image.fromarray(style_img_tranform)
style_img = loader(style_img_PIL).unsqueeze(0).to(device)

original_img_PIL = Image.fromarray(original_img)
original_img = loader(original_img_PIL).unsqueeze(0).to(device)

# Create generated image by clone content image or random
# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated = original_img.clone().requires_grad_(True)

model = VGG().to(device).eval()
# Hyperparameters
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

# Create dir to save image
# Get style name
style_name = getFileName(path_style)
path_dir_save_image = f'output-images/{style_name}/'
if os.path.exists(path_dir_save_image) == False:
    os.mkdir(path_dir_save_image)

# get file name
file_name = getFileName(path_img)

def cal_loss(gen_feature, orig_feature, style_feature):
    batch_size, channel, height, width = gen_feature.shape
    original_loss = torch.mean((gen_feature - orig_feature) ** 2)

    # Computer Gram Matrix
    G = gen_feature.view(channel, height * width).mm(
        gen_feature.view(channel, height * width).t()
    )

    A = style_feature.view(channel, height * width).mm(
        style_feature.view(channel, height * width).t()
    )
    style_loss = torch.mean((G - A) ** 2)
    return original_loss, style_loss

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0
    # original_loss, style_loss = cal_loss(generated_features, original_img_features, style_features)

    for gen_feature, orig_feature, style_feature in zip(generated_features,
                                            original_img_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature)**2)

        # Computer Gram Matrix
        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
        )
        style_loss += torch.mean((G -A)**2)


    total_loss = alpha*original_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 ==0:
        print(total_loss)
        path_save = os.path.join(path_dir_save_image, f'{file_name}_step_{step}.png')
        transforms_image = T.ToPILImage()
        image = transforms_image(generated[0])
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
        skimage.io.imsave(os.path.join(path_dir_save_image, f'{file_name}_step_{step}_1.png'), image)




