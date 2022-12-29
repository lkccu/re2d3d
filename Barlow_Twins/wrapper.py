#import pytorch_lightning as pl
from torch import nn
from torchvision.models.resnet import resnet50 as _resnet50
import torch
import fenc
from Barlow_Twins.loss import BarlowTwinsLoss

gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda' if gpus else 'cpu'

num_workers = 8
max_epochs = 800
knn_k = 200
knn_t = 0.1
classes = 10
batch_size = 512
seed=1

dataset_train_real = 0
dataset_train_aug = 0
train_loader = 0
class Projector(nn.Module):
    def __init__(self, in_features, out_features=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_features)
        )
    def forward(self, x):
        return self.fc(x)

class wrapper(nn.Module):

    def __init__(self):
        super(self).__init__()
        self.proj = Projector()

    def forward(x,self):
        #TODO:extract x,y from dataloader and encode'em
        x = self.proj(x)
        return x


params = {
        'device' : 'cuda' if gpus else 'cpu',
        'lambda_param' : 1e-4
       }

#loss_fn = BarlowTwinsLoss(**params)

#batch, dim_in, dim_h, dim_out = 128, 2000, 200, 20
#optimizer = torch.optim.Adam(wrapper.parameters(),lr=1e-4)

#max_epochs= 50
#for _ in range(0,max_epochs):
   # optimizer.zero_grad()
   # z_a,z_b = wrapper(train_loader)#trainLoader是在另一个文件中定义的
   # loss =loss_fn(z_a,z_b)
   # loss.backward()
   # optimizer.step()

#torch.save(wrapper.state_dict(),r'E:\course\31\machineLearning\AIforMedicine\re2D3D\Barlow_Twins\save_model')
#可以调用torch.load_state_dice(strict=false进行调用,map_location = device）是不是真的可以海待商榷
