import math
import os

import sys
import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(r'E:\course\31\machineLearning\AIforMedicine\re2D3D\Pointnet_Pointnet2_pytorch\log')
sys.path.append(r'E:\course\31\machineLearning\AIforMedicine\re2D3D\Pointnet_Pointnet2_pytorch\models')
from PIL import Image
from Pointnet_Pointnet2_pytorch.log.sem_seg.pointnet2_sem_seg.pointnet2_sem_seg import get_model
sys.path.append('/raft-pytorch')

import torch

import canny_edge_detector_3D.canny_edge_detector_3D as ced
import canny_edge_detector_3D.slicer_3D as slicer_3D
import numpy as np
import matplotlib.pyplot as plt
from raft1.train import sequence_loss
#-------------------------------------------------------------------------------------------------------------------
#------------------------3d canny detection-------------------------------------
def generate_contour(V,dod,width,length):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    detector,g = ced.cannyEdgeDetector3D(V, sigma=0.5, lowthresholdratio=0.35, highthresholdratio=0.55, weak_voxel=75,
                                       strong_voxel=256)
    img_edges = detector.detect()

    w = img_edges.X
    p = w.nonzero()#TODO 可能不是这么简单的nonzero

    return w,g,p
#------------------------------------------------------------------------------
"""
Euqation:WAdv = diag(W)b,
p p1:points pair, vector length:N_cp
w:contour points in 3d coordinate vector length:Ncp
g:contour gradient in 3d coordinate 
W:weight matrix a diag matrix of N_cp
"""
def PPCSolver(p,p1,w,g,W):
    n = np.dot(w,g,p1)
    n = np.linalg.norm(n)
    A = np.dot(n,w).T - n.T
    b = np.dot (n.T,w)
    W_inv =np.linalg.inv(W)
    dv = np.dot(W_inv,np.dot(A.T,A),A.T,b)

    return dv
#-----------------------------------------------------------------------------------------------------
#loss
# ---------------------------------------------------------------------------------------------
"""
the representation of the method DIRN 
"""

def loss_dirn(N_fl,N_cp,T_pred,T_gt,dv,W,f,f_gt,M_p):
    alpha =1
    gamma_ = 0.8
    beta = 0.5
    lambda_ = 1e-3
    zeta = 1e-5
    L_flow = sequence_loss(f,f_gt,gamma_)

    l1_T = np.sum(abs(T_pred - T_gt))
    l2_dv = math.sqrt(np.sum(np.pow(dv)))
    L_dirn = alpha * L_flow + beta * l1_T + lambda_ * l2_dv + zeta/2 + np.sum (np.pow(W))

    return L_dirn
# ---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
"""
3d canny detector
input: V(3d volume)
output: w g
"""
def canny_contour_detection(V):
    V = V[0]

    detector = ced.cannyEdgeDetector3D(V, sigma=0.5, lowthresholdratio=0.35, highthresholdratio=0.55, weak_voxel=75,
                                       strong_voxel=256)
    img_edges = detector.detect()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    plt.subplots_adjust(wspace=0.3)

    tracker1 = slicer_3D.Slicer3D(ax1, img_edges)

    w = tracker1.X
    g = np.gradient(tracker1.X)

    return w,g      #没有debug过，敬请debug一下
#---------------------------------------------------------------------------------------------
#raft

class MyDataset(Dataset):#简易型，只在这里面放入
    def __init__(self,flr_pth,drr_pth):
        self.flr_pth = flr_pth
        self.drr_pth = drr_pth

    def __len__(self):
        return len(os.listdir(self.flr_pth))

    def __getitem__(self,index):#TODO:图片类型应定义为F{index}.png /D{index}.png
        list_flr =os.listdir(self.flr_pth)
        list_flr =sorted(list_flr)
        list_drr =os.listdir(self.drr_pth)
        list_drr =sorted(list_drr)
        img_drr_pth = self.drr_pth + "D" + {index} +".jpg"
        img_flr_pth = self.flr_pth + "F" + {index} +".jpg"
        img_drr = np.array(Image.open(img_drr_pth))
        img_flr = np.array(Image.open((img_flr_pth)))
        return img_flr,img_drr
#TODO 这里的Image是Image类型的image嘛还是其它形式的image 倒是这里是肯定不需要一个dataloader形式的 但是这里连tensor形式的数组都不需要
img1 = 0
img2 = 0
#TODO 这里还需要完善

# -------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
"""
func:pointnet++
input:p,p1,w,g
output:W
我们现在要做的不是训练的过程，Ⅰ是用他现有的权重文件进行尝试，虽然效果会很差 Ⅱ训练过程可能需要单开一个文件进行尝试 可以先写用xxx.ckpt 后面再换成已经训练好的文件
"""
def pointnet(n, p1, w, g):
    xyz = concatate(n, p1, w, g)
    classifier = nn.Sigmoid()
    model = get_model(1)
    model.fc3 = classifier
    pointnet_pth = 0#TODO这里要改成权重文件的地址
    model = model.load_state_dict(torch.load(pointnet_pth),strict=False)
    W = model(xyz)

    return W

def concatate(n,p1,w,g):
    #TODO:CONCANATE THEM INTO A MATRIX(how） compute xyz
    xyz = 1
    return xyz

#--------------------------------------------------------------------------------------------
def pipeline():
    #TODO:produce a pipeline implementing DIRN.(NOT TRAIN)
    Tpred = 0
    return Tpred
  