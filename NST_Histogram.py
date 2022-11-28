import numpy as np
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
import cv2

# model = models.vgg19(pretrained=True).features
model = models.vgg19(pretrained=True)

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def getFileName(path):
    if '\\' in path:
        name = path.split('\\')[-1]
    else:
        name = path.split("/")[-1]
    file_name = os.path.splitext(name)[0]
    return file_name

def getContentSize(path_image):
    img = Image.open(path_image)
    img_copy = np.float32(img)
    h, w,_ = img_copy.shape
    return h, w



path_style = 'style-images/candy.jpg'
path_img = 'content-jpg/0003.jpg'


h, w = getContentSize(path_img)

device = torch.device("cuda" if torch.cuda.is_available else 'cpu')
image_size = 256

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)


img = Image.open(path_style).convert('RGB')
content = Image.open(path_img).convert('RGB')
original_img = load_image(path_img)
style_img_tranform = cal_mean_std.style_transformer2(content, img)


# convert from nparray to PIL
style_img_PIL = Image.fromarray(style_img_tranform)
style_img = loader(style_img_PIL).unsqueeze(0).to(device)

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

# Get style name
style_name = getFileName(path_style)
image_name = getFileName(path_img)

# Create dir to save image
path_dir_save_image = f'./output-images/{style_name}/{image_name}'
if os.path.exists(path_dir_save_image) == False:
    os.makedirs(path_dir_save_image)

# get file name
file_name = getFileName(path_img)

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0

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
        save_image(generated, os.path.join(path_dir_save_image, f'{file_name}_step_{step}.png'))
