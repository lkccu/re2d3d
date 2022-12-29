from typing import io
from PIL import Image
import numpy as np
import torch
from cv2.cv2 import resize
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torch.utils.data as data
import torch.nn
import random
import matplotlib.pyplot as plt

data_pth = r'E:\course\31\machineLearning\AIforMedicine\re2D3D\raft\drr_real\drr'
drr_pth = r'E:\course\31\machineLearning\AIforMedicine\re2D3D\raft\drr_real\drr\A001.png'
aug_pth = r'E:\course\31\machineLearning\AIforMedicine\re2D3D\raft\drr_real\style_aug\A002.png'

#%%


import numpy

drr =Image.open(drr_pth)
aug =Image.open(aug_pth)

fig1,(ax3,ax4) = plt.subplots(1,2,figsize =(256,256))
ax3.imshow(drr)
ax4.imshow(aug)
plt.show()
#%%
transforms.CenterCrop(drr,10)
#----------------------style augmentation----------------------
def add_noise(inputs,noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs)* noise_factor
    noisy = torch.clip(noisy,0.,1.)
    return noisy


transform = transforms.Compose([
            transforms.ColorJitter(0.5,0.5), #缩放图片保持长宽比不变,最短边为350像素
            transforms.RandomInvert(),
            transforms.ToTensor(), #将图片(Image)转成Tensor,归一化至(0,1)
            #需要对OpenCV读取的图像进行归一化处理才能与PIL读取的图像一致
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #常用标准化
 ])

def my_style_transforms(drr, aug):
    if random.random() > 0.5:
        drr = transform(drr)
        aug = transform(aug)
        elastic_transformer = transforms.ElasticTransform(alpha=250.0)
        drr = [elastic_transformer(drr) for _ in range(2)]
        #drr = add_noise(drr,1)
        #aug = add_noise(aug,0.5)
    # more transforms ...
    return drr, aug

drr,aug = my_style_transforms(drr,aug)

#drr = transforms.ToPILImage(drr)
#aug = transforms.ToPILImage(aug)

fig,(ax1,ax2) = plt.subplots(1,2,figsize =(256,256))
ax1.imshow(drr)
ax2.imshow(aug)
plt.show()
