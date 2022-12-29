from numpy import arange

import canny_edge_detector_3D as ced
import slicer_3D
import numpy as np
import matplotlib.pyplot as plt
from medpy.io.load import load

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img = load(r'E:\course\31\machineLearning\AIforMedicine\dataset\L1-L5FineSegMix-663case\101006073625_L2.nii.gz')
img = img[0]

detector = ced.cannyEdgeDetector3D(img, sigma=0.5, lowthresholdratio=0.35, highthresholdratio=0.55, weak_voxel=75, strong_voxel= 256)
img_edges = detector.detect()

fig, (ax1,ax2) = plt.subplots(1, 2)

plt.subplots_adjust(wspace=0.3)


tracker1 = slicer_3D.Slicer3D(ax1, img_edges)
tracker2 = slicer_3D.Slicer3D(ax2, img)

w = tracker1.X#TODO w和g是不是这么算的 可能不是需要改一下
g = np.gradient(tracker1.X)
# g需要重新计算一下
# w =[[[0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  ...,  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0]],, [[0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  ...,  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0],  [0 0 0 ... 0 0 0]],,
fig.canvas.mpl_connect('scroll_event', tracker1.on_scroll)
fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)

fig.set_figheight(8)
fig.set_figwidth(8)
plt.show()