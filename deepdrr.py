#! python3
"""Minimal projection example with DeepDRR."""
from matplotlib import pyplot as plt

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
import pickle


output_dir = test_utils.get_output_dir()
data_dir = "path/to/V" #todo 这里可以试下直接用cbct可以不可以
with open(data_dir,"rb") as f:
    V = pickle.load(f)
patient = deepdrr.Volume.from_parameters(
    data =V,
    materials = "bone"
)
patient.faceup()

carm = deepdrr.MobileCArm(isocenter=patient.center_in_world, alpha=0, beta=0, degrees=True)

with deepdrr.Projector(
    volume=patient,
    carm=carm,
    step=0.1,  # stepsize along projection ray, measured in voxels
    mode="linear",
    max_block_index=200,
    spectrum="90KV_AL40",
    photon_count=100000,
    add_scatter=False,
    threads=8,
    neglog=True,
) as projector:
    image = projector.project()
    plt.imsave("example.jpg",image)

