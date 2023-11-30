# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:35:15 2023

@author: User
"""
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import picos as pic
import scipy.io as scio
import math
import copy
from scipy.interpolate import interp1d
from scipy.interpolate import interpn
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment
from scipy import signal
from ipywidgets import interactive

class MyException(Exception):
    def __init__(self, input_catch):
        self.catch = str(input_catch)

class DataError(MyException):
    def __str__(self):
        return "error with data: "

class ArgError(MyException):
    def __str__(self):
        return "error with argument: "

class ProcessingFuncs:
    """
    Collection of static methods that may be used in other methods.\n
    """
    def __init__(self):
        pass

    @staticmethod
    def smooth_csi(input_csi, rx=2, sub=15):

        nrx = input_csi.shape[1]
        nsub = input_csi.shape[0]

        input_csi = input_csi.swapaxes(0, 1)

        output = [input_csi[i:i + rx, j:j + sub].reshape(-1)
                  for i in range(nrx - rx + 1)
                  for j in range(nsub - sub + 1)]

        return np.array(output)

    @staticmethod
    def noise_space(input_csi):
        """
        Calculates self-correlation and eigen vectors of given csi.\n
        For AoA, input CSI of (nsub, nrx).\n
        For ToF and Doppler, input CSI of (nrx, nsub).\n
        :param input_csi: complex csi
        :return: noise space vectors
        """

        input_csi = np.squeeze(input_csi)

        value, vector = np.linalg.eigh(input_csi.T.dot(np.conjugate(input_csi)))
        descend_order_index = np.argsort(-value)
        vector = vector[:, descend_order_index]
        noise_space = vector[:, 1:]

        return noise_space

    @staticmethod
    def conjmul_dynamic(input_csi, ref, reference_antenna, subtract_mean=True):
        if ref == 'rx':
            hc = input_csi * input_csi[:, :, reference_antenna, :][..., np.newaxis, :].repeat(3, axis=2).conj()
        elif ref == 'tx':
            hc = input_csi * input_csi[:, :, :, reference_antenna][..., np.newaxis].repeat(3, axis=3).conj()

        if subtract_mean is True:
            static = np.mean(hc, axis=0)
            dynamic = hc - static
        else:
            dynamic = hc
        return dynamic

    @staticmethod
    def divison_dynamic(input_csi, ref, reference_antenna, subtract_mean=True):

        re_csi = (np.abs(input_csi) + 1.e-6) * np.exp(1.j * np.angle(input_csi))
        if ref == 'rx':
            hc = input_csi / re_csi[:, :, reference_antenna, :][..., np.newaxis, :].repeat(3, axis=2)
        elif ref == 'tx':
            hc = input_csi / re_csi[:, :, :, reference_antenna][..., np.newaxis].repeat(3, axis=3)

        if subtract_mean is True:
            static = np.mean(hc, axis=0)
            dynamic = hc - static
        else:
            dynamic = hc
        return dynamic

    @staticmethod
    def highpass(fs=1000, cutoff=2, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def aoa_doppler_by_music(input_csi=None,
                             window_length=None,
                             stride=None,
                             timestamps=None,
                             raw_window=True):

        lightspeed = 299792458
        center_freq = 5.89 * 1e+09
        bandwidth = 20
        sampling_rate = 1000
        dist_antenna = lightspeed / center_freq / 2.
        torad = np.pi / 180
        ntx = 1
        nrx = 3
        nsub = 30
        input_theta_list=np.arange(-90, 91, 1.)
        input_velocity_list=np.arange(-5, 5.05, 0.05)
        length = len(input_csi)
        noise = ProcessingFuncs.noise_space
        dynamic = ProcessingFuncs.conjmul_dynamic
        

        print("AoA-Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if input_csi is None:
                raise DataError("amplitude: " + str(input_csi) + "\nPlease load data")

            # Each window has ts of packets (1 / sampling_rate * window_length = t)
            delay_list = np.arange(0, window_length, 1.).reshape(-1, 1) / sampling_rate
            
            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            
            velocity_list = np.array(input_velocity_list).reshape(-1, 1)

            steering_aoa = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                antenna_list.T) * center_freq / lightspeed).reshape(-1, 1)
            
            spectrum = np.zeros(((length - window_length) // stride, len(input_theta_list),
                                 len(input_velocity_list)))
            
            temp_timestamps = np.zeros((length - window_length) // stride)

            # Using windowed dynamic extraction
            for i in range((length - window_length) // stride):

                csi_windowed = input_csi[i * stride: i * stride + window_length]

                if raw_window is True:
                    
                    noise_space = noise(csi_windowed.swapaxes(0, 1).reshape(nsub, window_length * nrx))
                    
                else:
                    # Using windowed dynamic extraction
                    csi_dynamic = dynamic(csi_windowed, ref='rx', reference_antenna=1)
                    noise_space = noise(csi_dynamic.swapaxes(0, 1).reshape(nsub, window_length * nrx))

                if timestamps is True:
                    
                    # Using original timestamps (possibly uneven intervals)
                    delay_list = timestamps[i * stride: i * stride + window_length] - \
                                 timestamps[i * stride]

                for j, velocity in enumerate(velocity_list):

                    steering_doppler = np.exp(-1.j * 2 * np.pi * center_freq * delay_list * velocity /
                                              lightspeed).reshape(-1, 1)
                    steering_vector = steering_doppler.dot(steering_aoa.T
                                                           ).reshape(len(delay_list), len(input_theta_list), nrx)
                    steering_vector = steering_vector.swapaxes(0, 1
                                                               ).reshape(len(input_theta_list), nrx * len(delay_list))

                    a_en = np.conjugate(steering_vector).dot(noise_space)
                    spectrum[i, :, j] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            spectrum = np.log(spectrum)
            
            #self.viewer = AoADopplerViewer(name=self.name, spectrum=self.spectrum, timestamps=temp_timestamps)
            print("AoA-Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")
        
        return spectrum 

    def interpratation_csi_data(csi_data, timestamps, time_steps=0.5):
        # 得到插值后的时间点
        new_timestamps = np.arange(timestamps[0], timestamps[-1] + time_steps, time_steps)
    
        # 用时间戳对CSI数据排序
        csi_data = csi_data[np.argsort(timestamps)]
        timestamps.sort()
    
        # 创建储存插值结果的数组
        interp_data = np.empty((len(new_timestamps),) + csi_data.shape[1:], dtype=np.complex)
    
        # 拆分成实部和虚部，分别对它们插值
        for i in range(csi_data.shape[1]):
            for j in range(csi_data.shape[2]):
                for k in range(csi_data.shape[3]):
                    # 拆分实部和虚部
                    csi_real = np.real(csi_data[:, i, j, k])
                    csi_imag = np.imag(csi_data[:, i, j, k])
    
                    # 创建插值函数
                    interp_func_real = RegularGridInterpolator((timestamps,), csi_real, bounds_error=False, fill_value=None)
                    interp_func_imag = RegularGridInterpolator((timestamps,), csi_imag, bounds_error=False, fill_value=None)
    
                    # 在新的时间点上进行插值
                    interp_real = interp_func_real(new_timestamps[:, None])
                    interp_imag = interp_func_imag(new_timestamps[:, None])
    
                    # 合并实部和虚部为复数，储存起来
                    interp_data[:, i, j, k] = interp_real + 1j * interp_imag
    
        return interp_data


    def sanitize_phase(input_csi=None):
        
        # input_csi = csi_matrix
        nrx = 3
        nsub = 30
        length = len(input_csi)

        print("apply SpotFi Algorithm1 to remove STO...", end='')

        try:
            if input_csi is None:
                raise DataError("phase")
            #input_csi = csi_matrix
            fit_x = np.concatenate([np.arange(0, nsub) for _ in range(nrx)])
            fit_y = np.unwrap(np.angle(np.squeeze(input_csi)), axis=1).swapaxes(1, 2).reshape(length, -1)

            a = np.stack((fit_x, np.ones_like(fit_x)), axis=-1)
            fit = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(fit_y.T).T
            # fit = np.array([np.polyfit(fit_x, fit_y[i], 1) for i in range(self.data.length)])

            phase = np.unwrap(np.angle(input_csi), axis=1) - np.arange(nsub).reshape(
                (1, nsub, 1, 1)) * fit[:, 0].reshape(length, 1, 1, 1)
            print("Done")

            sanitize_phase_csi = np.abs(input_csi) * np.exp(1.j * phase)

        except DataError as e:
            print(e, "\nPlease load data")
        return sanitize_phase_csi
    
    def csi_sanitization(csi_data, M, N):
    # M = 3
    # N = 30
        freq_delta = 2 * 312.5e3

        csi_phase = np.zeros(M*N)
        for ii in range(1, M+1):
            if ii == 1:
                csi_phase[(ii-1)*N: ii*N] = np.unwrap(np.angle(csi_data[(ii-1)*N: ii*N]))
            else:
                csi_diff = np.angle(np.multiply(csi_data[(ii-1)*N: ii*N], np.conj(csi_data[(ii-2)*N: (ii-1)*N])))
                csi_phase[(ii-1)*N: ii*N] = np.unwrap(csi_phase[(ii-2)*N: (ii-1)*N]+csi_diff)

        ai = 2 * np.pi * freq_delta * np.tile(np.arange(0, N),M).reshape([1, N*M])
        bi = np.ones((1, len(csi_phase)))
        ci = csi_phase
        A = np.dot(ai, np.transpose(ai))[0]
        B = np.dot(ai, np.transpose(bi))[0]
        C = np.dot(bi, np.transpose(bi))[0]
        D = np.dot(ai, np.transpose(ci))[0]
        E = np.dot(bi, np.transpose(ci))[0]
        rho_opt = (B * E - C * D) / (A * C - np.square(B))
        beta_opt = (B * D - A * E) / (A * C - np.square(B))

        csi_phase_2 = csi_phase + 2 * np.pi * freq_delta * np.tile(np.arange(0, N),M).reshape([1, N*M]) * rho_opt + beta_opt
        result = np.multiply(np.abs(csi_data), np.exp(1j * csi_phase_2))
        return result
    
    def interpratation_csi_data(csi_data, timestamps, time_steps=0.05):
        # 得到插值后的时间点
        # csi_data = csi_matrix
        new_timestamps = np.arange(timestamps[0], timestamps[-1] + time_steps, time_steps)
    
        # 用时间戳对CSI数据排序
        csi_data = csi_data[np.argsort(timestamps)]
        timestamps.sort()
    
        # 创建储存插值结果的数组
        interp_data = np.empty((len(new_timestamps),) + csi_data.shape[1:], dtype=np.complex128)
    
        # 拆分成实部和虚部，分别对它们插值
        for i in range(csi_data.shape[1]):
            for j in range(csi_data.shape[2]):
                for k in range(csi_data.shape[3]):
                    # 拆分实部和虚部
                    csi_real = np.real(csi_data[:, i, j, k])
                    csi_imag = np.imag(csi_data[:, i, j, k])
    
                    # 创建插值函数
                    interp_func_real = RegularGridInterpolator((timestamps,), csi_real, bounds_error=False, fill_value=None)
                    interp_func_imag = RegularGridInterpolator((timestamps,), csi_imag, bounds_error=False, fill_value=None)
    
                    # 在新的时间点上进行插值
                    interp_real = interp_func_real(new_timestamps[:, None])
                    interp_imag = interp_func_imag(new_timestamps[:, None])
    
                    # 合并实部和虚部为复数，储存起来
                    interp_data[:, i, j, k] = interp_real + 1j * interp_imag
    
        return interp_data
    
    def plot_func(aoa_doppler_map):
        windows = aoa_doppler_map.shape[0]-1
        plt.imshow(aoa_doppler_map[windows,:,:], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Amplitude')
        #plt.xticks(np.arange(-90, 91, 1.))
        #plt.yticks(np.arange(-5, 5.05, 0.05))
        plt.xlabel('AoA')
        plt.ylabel('DOPPLER')
        plt.show()

    def verbose_series(csi_data, start=0, end=None, sub=14, rx=0, tx=0, notion=''):

        if end is None:
            end = len(csi_data)

        plt.plot(np.unwrap(np.angle(csi_data[start:end, sub, rx, tx]), axis=0))
        plt.grid()
        plt.title(notion)
        plt.xlabel("packet")
        plt.ylabel("phase")
        plt.show()

    def show_antenna_strength(input_csi):

        try:
            if input_csi is None:
                raise DataError("csi")

        except DataError as e:
            print(e, "\nPlease load data")

        else:
            mean_abs = np.mean(np.abs(input_csi), axis=(0, 1))
            return mean_abs

    def doppler_by_music(input_csi, timestamps=None, input_velocity_list=np.arange(-5, 5.01, 0.01),
                         window_length=100,
                         stride=100,
                         pick_rx=0,
                         pick_tx=0,
                         ref_antenna=1,
                         dynamic=True):

        sampling_rate = 1000
        noise = ProcessingFuncs.noise_space
        wdyn = ProcessingFuncs.conjmul_dynamic
        length = len(input_csi)
        lightspeed = 299792458
        center_freq = 5.89 * 1e+09
        bandwidth = 20
        dist_antenna = lightspeed / center_freq / 2.
        torad = np.pi / 180
        ntx = 1
        nrx = 3
        nsub = 30

        print("Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if input_csi is None:
                raise DataError("amplitude: " + "\nPlease load data")

            # Each window has (window_length / sampling_rate) seconds of packets
            delay_list = np.arange(0, window_length, 1.).reshape(-1, 1) / sampling_rate
            velocity_list = np.array(input_velocity_list[::-1]).reshape(-1, 1)
            total_strides = (length - window_length) // stride

            if pick_rx == 'strong':
                pick_rx = np.argmax(ProcessingFuncs.show_antenna_strength(input_csi))
            elif pick_rx == 'weak':
                pick_rx = np.argmin(ProcessingFuncs.show_antenna_strength(input_csi))

            spectrum = np.zeros((len(input_velocity_list), total_strides))
            temp_timestamps = np.zeros(total_strides)

            for i in range(total_strides):

                csi_windowed = input_csi[i * stride: i * stride + window_length]

                if dynamic is True:
                    # Using windowed dynamic extraction
                    csi_dynamic = wdyn(csi_windowed, ref='rx', reference_antenna=ref_antenna)
                    noise_space = noise(csi_dynamic[:, :, pick_rx, pick_tx].T)
                else:
                    noise_space = noise(csi_windowed[:, :, pick_rx, pick_tx].T)

                if timestamps is True:
                    # Using original timestamps (possibly uneven intervals)
                    delay_list = timestamps[i * stride: i * stride + window_length] - \
                                 timestamps[i * stride]

                steering_vector = np.exp(-1.j * 2 * np.pi * center_freq * velocity_list.dot(delay_list.T) / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

                temp_timestamps[i] = timestamps[i * stride]

            #self.spectrum = np.log(spectrum)
            #self.viewer = DopplerViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps, xlabels=temp_timestamps)

            print("Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def aoa_tof_by_music(input_csi, input_theta_list=np.arange(-90, 91, 1.),
                         input_dt_list=np.arange(-1.e-7, 2.e-7, 1.e-9),
                         smooth=False):
        sampling_rate = 1000
        noise = ProcessingFuncs.noise_space
        wdyn = ProcessingFuncs.conjmul_dynamic
        length = len(input_csi)
        lightspeed = 299792458
        center_freq = 5.89 * 1e+09
        delta_subfreq = 3.125e+05
        bandwidth = 20
        nrx = 3
        ntx = 1
        nsub = 30
        dist_antenna = lightspeed / center_freq / 2.
        torad = np.pi / 180
        smoothing = ProcessingFuncs.smooth_csi
        noise = ProcessingFuncs.noise_space
        subfreq_list = np.arange(center_freq - 28 * delta_subfreq, center_freq + 32 * delta_subfreq, 2 * delta_subfreq).reshape(-1, 1)

        print("AoA-ToF by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if input_csi is None:
                raise DataError("amplitude: " +"\nPlease load data")

            if smooth not in (True, False):
                raise ArgError("smooth:" + str(smooth))

            if smooth is True:
                print("apply Smoothing via SpotFi...")

            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            dt_list = np.array(input_dt_list).reshape(-1, 1)

            steering_aoa = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                            antenna_list.T) * center_freq / lightspeed).reshape(-1, 1)

            spectrum = np.zeros((length, len(input_theta_list), len(input_dt_list)))

            for i in range(length):

                if smooth is True:
                    pass

                noise_space = noise(input_csi[i].reshape(1, -1))   # nrx * nsub columns

                for j, tof in enumerate(dt_list):

                    if smooth is True:
                        steering_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                            antenna_list[:2].dot(subfreq_list[:15])) / lightspeed)
                    else:
                        steering_tof = np.exp(-1.j * 2 * np.pi * subfreq_list * tof).reshape(-1, 1)
                        steering_vector = steering_tof.dot(steering_aoa.T).reshape(nsub, len(input_theta_list), nrx)
                        steering_vector = steering_vector.swapaxes(0, 1).reshape(len(input_theta_list), nrx * nsub)

                    a_en = np.conjugate(steering_vector).dot(noise_space)
                    spectrum[i, :, j] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            #self.spectrum = np.log(spectrum)
            #self.viewer = AoAToFViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print("AoA-ToF by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")

    def remove_ipo(input_csi, reference_antenna=0, cal_dict=None):
        sampling_rate = 1000
        noise = ProcessingFuncs.noise_space
        wdyn = ProcessingFuncs.conjmul_dynamic
        length = len(input_csi)
        lightspeed = 299792458
        center_freq = 5.89 * 1e+09
        delta_subfreq = 3.125e+05
        bandwidth = 20
        nrx = 3
        ntx = 1
        nsub = 30
        distance_antenna = lightspeed / center_freq / 2.
        torad = np.pi / 180

        print("apply phase calibration according to", str(cal_dict.keys())[10:-1], "...", end='')

        try:
            if input_csi is None:
                raise DataError("csi")

            if reference_antenna not in (0, 1, 2):
                raise ArgError("reference_antenna")

            if cal_dict is None:
                raise DataError("reference")

            ipo = []
            # cal_dict: "{'xx': MyCsi}"

            for key, value in cal_dict.items():

                ref_angle = eval(key)

                ref_csi = value.csi
                ref_diff = np.mean(ref_csi * ref_csi[:, :, reference_antenna][:, :, np.newaxis].conj(),
                                   axis=(0, 1))
                true_diff = np.exp([-1.j * 2 * np.pi * distance_antenna * antenna * center_freq * np.sin(
                    ref_angle * torad) / lightspeed for antenna in range(nrx)]).reshape(-1, 1)

                ipo.append(ref_diff.reshape(-1, 1) * true_diff.conj())

            ipo = np.squeeze(np.mean(ipo, axis=0))

            remove_ipo_csi = input_csi * ipo[np.newaxis, np.newaxis, :, np.newaxis].conj()

            print("Done")
            
            return remove_ipo_csi

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify an integer from 0~2")

    def csi_highpass_filter(input_csi):
        order = 10
        cutoff_freq = 200
        nsub = 30
        nrx = 3
        ntx = 1
        b, a = signal.butter(order, cutoff_freq, fs=1000, btype='low')
        for sub in range(nsub):
            for rx in range(nrx):
                for tx in range(ntx):
                    input_csi[:, sub, rx, tx] = signal.butter(b, a, input_csi[:, sub, rx, tx])
        return input_csi
    
    def optimal_initialization(G, L): # G=6,L=2
        prob = pic.Problem()
        variables = prob.add_variable('np.variables', G*G*L*L, vtype='binary')
        IJ = []
        IJK = []
        PQ = []
    
        for i in range(G): #G*L
            for j in range(L):
                IJ.append((i, j))
    
        for i in range(G): #G*L
            for j in range(G):
                for k in range(L):
                    IJK.append((i, j, k))
    
        for j in range(G):
            for l in range(L):
                PQ.append((j, l))
    
        # M[i,j,k,l] = M[G*L*L*ii + L*L*jj + L*kk + ll]
        # constraints = [constraints np.variables(ii,ii,jj,jj) == 1]
        prob.add_list_of_constraints([variables[G*L*L*ii + L*L*ii + L*jj + jj] == 1 for (ii, jj) in IJ])
    
        for ii in range(G):
            for jj in range(L):
                # constraints = [constraints np.squeeze(np.variables(ii,:,jj,:)) == np.squeeze(np.variables(:,ii,:,jj))];
                prob.add_list_of_constraints([variables[G*L*L*ii + L*L*kk + L*jj + ll] == variables[G*L*L*kk + L*L*ii + L*ll + jj] for (kk, ll) in IJ])
    
        # constraints = [constraints np.sum(np.sum(np.sum(np.variables, 2), 3), 4) == G * L]
        prob.add_list_of_constraints([pic.sum(variables[G*L*L*ii: G*L*L*(ii+1)]) == G*L for ii in range(G)])
    
        # constraints = [constraints np.sum(np.variables,4) == 1]
        prob.add_list_of_constraints([pic.sum(variables[G*L*L*ii + L*L*jj + L*kk: G*L*L*ii + L*L*jj + L*(kk+1)]) == 1 for (ii, jj, kk) in IJK])
    
        for ii in range(G):
            for jj in range(G):
                if ii == jj:
                    continue
                for mm in range(L):
                    for nn in range(L):
                        # constraints = [constraints np.variables(ii,jj,mm,nn) + np.squeeze(np.variables(jj,:,nn,:)) <= 1 + np.squeeze(np.variables(ii,:,mm,:))];
                        prob.add_list_of_constraints([variables[G*L*L*ii + L*L*jj + L*mm + nn] + variables[G*L*L*jj + L*L*pp + L*nn + qq] <= \
                                                      1 + variables[G*L*L*ii + L*L*pp + L*mm + qq] for (pp, qq) in PQ])
    
        return prob, variables
    
    
    def matching(estimated_parameter, optimize_variables, optimize_problem):
        L = np.size(estimated_parameter, 1)
        G = np.size(estimated_parameter, 2)
        estimated_parameter_temp = copy.copy(estimated_parameter)
    
        cost_matrix = np.ones([G, G, L, L]) * 10000
        temp_parameter = np.zeros([np.size(estimated_parameter, 0), L], dtype=complex)
        for ii in range(G):
            for kk in range(L):
                cost_matrix[ii, ii, kk, kk] = 0
                for qq in range(L):
                        temp_parameter[:, qq] = copy.copy(estimated_parameter[:, kk, ii])
                for jj in range(ii+1, G):
                    cost_matrix[ii, jj, kk, :] = np.sqrt(np.sum(np.square(np.abs(temp_parameter - estimated_parameter[:, :, jj])), 0))
                    cost_matrix[jj, ii, :, kk] = np.sqrt(np.sum(np.square(np.abs(temp_parameter - estimated_parameter[:, :, jj])), 0))
    
        cost_matrix_pic = pic.new_param('cost_matrix', np.reshape(cost_matrix, [G*G*L*L, 1]))
        optimize_problem.set_objective('min', cost_matrix_pic | optimize_variables)
        optimize_problem.solve(verbose=0)
        # print(optimize_problem)
    
        edges = optimize_variables.value
        edges = [0 if x<=1e-5 else 1 for x in edges]
        estimated_index = np.zeros([L, G], dtype=int)
        estimated_index[:, 0] = np.arange(0, L)
        for ii in range(1, G):
            for jj in range(L):
                estimated_index[jj, ii] = np.flatnonzero(edges[L*L*ii + L*jj: L*L*ii + L*(jj+1)])  # 找到非0索引
            estimated_parameter_temp[:, :, ii] = estimated_parameter_temp[:, estimated_index[:, ii], ii]
        return estimated_parameter_temp, estimated_index
    
        