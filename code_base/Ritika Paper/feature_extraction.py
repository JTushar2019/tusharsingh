import numpy as np
import pandas as pd
import mne ,os, emd
from scipy import signal
from scipy import stats as st
import neurokit2 as nk
from tqdm import tqdm
from numba import jit


data = np.load('/home/tusharsingh/DATAs/sub_data.npz')
sub_eog, sub_label = data['a'], data['b']

data = np.load('/home/tusharsingh/DATAs/pat_data.npz')
pat_eog, pat_label = data['a'], data['b']

def feature_maker(X):
    imf = emd.sift.ensemble_sift(X, max_imfs=5, nensembles=24, nprocesses=6, ensemble_noise=0.2)
    imf = imf.T
    imf = np.vstack((imf, X - np.sum(imf, axis = 0)))
    IP, IF, IA = emd.spectra.frequency_transform(imf, 200, 'hilbert')
    f, hht = emd.spectra.hilberthuang(IF, IA, sample_rate=200)
    approx_entropy = nk.entropy_approximate(X)[0]
    del(IA)
    IMF_mean = np.mean(imf, axis = 1)
    IMF_mode = st.mode(imf, axis = 1)[0].flatten()
    IMF_sdiv = st.tstd(imf, axis = 1)
    IMF_kurt = st.kurtosis(imf, axis = 1)
    IMF_skew = st.skew(imf, axis = 1)
    feature = np.hstack([IMF_mean, IMF_mode, IMF_sdiv, IMF_kurt, IMF_skew])

    IF_mean = np.mean(IF,axis = 1)
    IF_median = np.median(IF, axis = 1)
    IF_variance = np.var(IF, axis = 1)
    IF_counChange = np.sum((np.diff(IF, axis = 1) != 0), axis = 1)
    IE_median = np.median(IP, axis = 1)

    feature = np.hstack([feature,IF_median, IF_mean, IF_variance, IF_counChange, IE_median])

    total_power = np.sum(hht[(0.5 <= f) & (f <= 49.5)])
    # relative power features
    theta = np.sum(hht[(3.5 <= f) & (f <= 7.5)]) / total_power
    alpha = np.sum(hht[(8.5 <= f) & (f <= 11)]) / total_power
    beta = np.sum(hht[(15.5 <= f) & (f <= 30)]) / total_power
    gamma = np.sum(hht[(30 <= f) & (f <= 49.5)]) / total_power
    delta = np.sum(hht[(0.5 <= f) & (f <= 3.5)]) / total_power
    k_complex_spindle = (np.sum(hht[(8.5 <= f) & (f <= 1.5)]) + \
        np.sum(hht[(11 <= f) & (f <= 15)])) / total_power
    alpha2theta = alpha / theta

    Relative = np.array([alpha, beta, gamma, delta, k_complex_spindle, alpha2theta,approx_entropy])
    feature = np.hstack([feature, Relative])
    return feature[:]


def linear_process(X):
    final_feature = np.zeros((X.shape[0],67))
    for i in range(X.shape[0]):
        final_feature[i] = feature_maker(X[i])
    return final_feature

import concurrent.futures
with concurrent.futures.ProcessPoolExecutor() as executor:
    a = executor.submit(linear_process, pat_eog[:10000,:])
    b = executor.submit(linear_process, pat_eog[10000:20000,:])
    c = executor.submit(linear_process, pat_eog[20000:30000,:])
    d = executor.submit(linear_process, pat_eog[30000:,:])

    e = executor.submit(linear_process, sub_eog[:10000,:])
    f = executor.submit(linear_process, sub_eog[10000:20000:,:])
    g = executor.submit(linear_process, sub_eog[20000:,:])

    pat_67 = np.vstack((a.result(), b.result(), c.result(), d.result()))
    sub_67 = np.vstack((e.result(), f.result(), g.result()))

np.savez('/home/tusharsingh/DATAs/pat_data_67_new', a=pat_67, b=pat_label)
np.savez('/home/tusharsingh/DATAs/sub_data_67_new', a=sub_67, b=sub_label)