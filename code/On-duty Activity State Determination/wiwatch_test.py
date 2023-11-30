# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:08:49 2023

@author: User
"""
from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel

from CSIKit.reader import get_reader
from CSIKit.tools.batch_graph import BatchGraph
from CSIKit.util import csitools

from ipywidgets import interactive

import seaborn
import numpy as np
import wiwatch_funcs
import time

import matplotlib.pyplot as plt

my_reader = get_reader("D:\\MATLAB\\CSI-python\\data\\jiashishi_s01.dat")
csi_data = my_reader.read_file("D:\\MATLAB\\CSI-python\\data\\jiashishi_s01.dat", scaled=True)
csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="complex")
csi_matrix = csi_matrix[:, :, :, 0:1]
timestamps = csi_data.timestamps

## CSI interpration
start_time = time.time()
csi_matrix =  wiwatch_funcs.ProcessingFuncs.interpratation_csi_data(csi_matrix,timestamps, 0.001)
end_time = time.time()
execution_time = end_time - start_time
print("CSI interpration execution time：",execution_time)
##

## CSI phase sanitization
start_time = time.time()
csi_matrix = wiwatch_funcs.ProcessingFuncs.sanitize_phase(csi_matrix)
end_time = time.time()
execution_time = end_time - start_time
print("CSI sanitize运行时长：",execution_time)

raw_csi_phase = np.angle(csi_matrix[:,15,1,:])
plt.plot(raw_csi_phase)
#sanitized_csi_phase = np.angle(sanitize_phase_csi[:,15,1,:])
#plt.plot(sanitized_csi_phase)

## CSI low pass filtering
start_time = time.time()
csi_matrix = wiwatch_funcs.ProcessingFuncs.csi_highpass_filter(csi_matrix)
end_time = time.time()
execution_time = end_time - start_time
print("CSI lowpass filter execution time：",execution_time)
##


## AoA-Doppler estimation
start_time = time.time()
#  csi_matrix = csi_matrix[0:100,:,:,:]
aoa_doppler_spec = wiwatch_funcs.ProcessingFuncs.aoa_doppler_by_music(csi_matrix, 100, 100, timestamps)
end_time = time.time()
execution_time = end_time - start_time
print("AoA-Doppler estimation execution time：",execution_time)
interactive_plot = interactive(wiwatch_funcs.ProcessingFuncs.plot_func(aoa_doppler_spec), window=(0, aoa_doppler_spec.shape[0]))
interactive_plot

aoa_matrix = aoa_doppler_spec[:,:]
plt.plot(aoa_matrix)
plt.show()


## parametor matching




## 



