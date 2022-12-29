import os
from typing import io

import numpy as np
import torch
from cv2.cv2 import resize
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from pathlib import Path
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torch.utils.data as data
import torch.nn
import torchvision.transforms.functional as TF
import random
import skimage.io as io
from sklearn.model_selection import train_test_split



data_pth = r'E:\course\31\machineLearning\AIforMedicine\re2D3D\raft\drr_real\drr'
drr_pth = r'E:\course\31\machineLearning\AIforMedicine\re2D3D\raft\drr_real\drr'
aug_pth = r'E:\course\31\machineLearning\AIforMedicine\re2D3D\raft\drr_real\style_aug'



#----------------------style augmentation----------------------
def add_noise(inputs,noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs)* noise_factor
    noisy = torch.clip(noisy,0.,1.)
    return noisy


transform = transforms.Compose([
            transforms.ColorJitter(0.5,0.5), #缩放图片保持长宽比不变,最短边为350像素
            transforms.RandomInvert(), #从图片中间切出320*320的图片
            transforms.ToTensor(), #将图片(Image)转成Tensor,归一化至(0,1)
            #需要对OpenCV读取的图像进行归一化处理才能与PIL读取的图像一致
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #常用标准化
 ])

def my_style_transforms(drr, aug):
    if random.random() > 0.5:
        drr = transform(drr)
        aug = transform(aug)
        drr =add_noise(drr,)
    # more transforms ...
    return drr, aug
#:return a PIL image.
#TODO:将aug之后的文件保存到源文件夹下
#-----------------------load dataset------------------



class MyDataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self,list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self,index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        fixed_image = torch.Tensor(
            resize(io.imread(drr_pth + ID + '_1.jpg'), (480, 480, 3)))
        moving_image = torch.Tensor(
            resize(io.imread(aug_pth + ID + '_2.jpg'), (480, 480, 3)))
        return fixed_image, moving_image

params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6,
          'worker_init_fn': np.random.seed(42)
        }
filename = list(set([x.split('_')[0]
                         for x in os.listdir('./fire-fundus-image-registration-dataset/')]))#这里的保存还没有改完
partition ={}
partition['train'], partition['validation'] = train_test_split(filename, test_size=0.33, random_state=42)

train_dataset = Dataset(partition['train'])
training_generator = data.DataLoader(train_dataset, **params)
