from collections import OrderedDict
import torch.nn.functional as F

import torch.nn as nn
from Pointnet_Pointnet2_pytorch.models.pointnet2_cls_msg import get_model
classifier = nn.Sigmoid()
model = get_model(1)
model.fc3 = classifier
print(model)