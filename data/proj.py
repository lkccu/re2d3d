# %% Usage of FDK

# the FDK algorithm has been taken and modified from
# 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction Matlab examples
# https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct--cbct--projection-backprojection-fdk--iterative-reconstruction-matlab-examples

# The algorithm takes, as eny of them, 3 mandatory inputs:
# PROJECTIONS: Projection data
# GEOMETRY   : Geometry describing the system
# ANGLES     : Propjection angles
# And has a single optional argument:
# FILTER: filter type applied to the projections. Possible options are
#        'ram_lak' (default)
#        'shepp_logan'
#        'cosine'
#        'hamming'
#        'hann'
# The choice of filter will modify the noise and sopme discreatization
# errors, depending on which is chosen.
#
##% Demo 6: Algorithms01
#
# In this demo the usage of the FDK is explained
#
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This file is part of the TIGRE Toolbox
#
# Copyright (c) 2015, University of Bath and
#                     CERN-European Organization for Nuclear Research
#                     All rights reserved.
#
# License:            Open Source under BSD.
#                     See the full license at
#                     https://github.com/CERN/TIGRE/blob/master/LICENSE
#
# Contact:            tigre.toolbox@gmail.com
# Codes:              https://github.com/CERN/TIGRE/
# Coded by:           Ander Biguri
# --------------------------------------------------------------------------
#%%Initialize
import tigre
import numpy as np
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
import nrrd
import nibabel as nib
#%% Geometry

#%%
#Geometry settings
geo = tigre.geometry(mode='cone',  default=False)
geo.DSD = 1500                                     # Distance Source Detector      (mm)
geo.DSO = 1000                                     # Distance Source Origin        (mm)
# Detector parameters
geo.nDetector = np.array((300, 300))               # number of pixels              (px)
geo.dDetector = np.array((0.8, 0.8))               # size of each pixel            (mm)
geo.sDetector = geo.nDetector * geo.dDetector    # total size of the detector    (mm)
# Image parameters
geo.nVoxel = np.array((1024, 1024, 390))             # number of voxels              (vx)
geo.sVoxel = np.array((128, 128, 128))             # total size of the image       (mm)
geo.dVoxel = geo.sVoxel/geo.nVoxel               # size of each voxel            (mm)
# Offsets
geo.offOrigin = np.array((0, 0, 0))                # Offset of image from origin   (mm)
geo.offDetector = np.array((0, 0))                 # Offset of Detector            (mm)
#%%

#%% Load data and generate projections
# define angles
angles = np.linspace(0, 2 * np.pi, 100)
# Load thorax phatom data
pth = r'/root/TIGRE/Python/9.nii.gz'
cbct = nib.load(pth)
cbct = cbct.get_fdata(dtype = 'float32')
# generate projections
projections = tigre.Ax(cbct, geo, angles)
tigre.plotProj(projections)
# add noise
#noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))

#%%

imgFDK1 = algs.fdk(projections, geo, angles, filter="hann")
imgFDK2 = algs.fdk(projections, geo, angles, filter="ram_lak")

# They look quite the same
tigre.plotimg(np.concatenate([imgFDK1, imgFDK2], axis=1), dim="Z" )

# but it can be seen that one has bigger errors in the whole image, while
# hte other just in the boundaries
tigre.plotimg(np.concatenate([abs(cbct - imgFDK1), abs(cbct - imgFDK2)], axis=1), dim="Z")
