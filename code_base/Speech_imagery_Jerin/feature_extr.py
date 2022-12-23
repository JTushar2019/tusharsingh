import scipy.io as sio
import numpy as np
import scipy
import os
import re
import random
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

folder_path = {"Long_words": "/home/tusharsingh/DATAs/speech_EEG/Long_words",
               "Short_Long_words": "/home/tusharsingh/DATAs/speech_EEG/Short_Long_words",
               "Short_words": "/home/tusharsingh/DATAs/speech_EEG/Short_words",
               "Vowels": "/home/tusharsingh/DATAs/speech_EEG/Vowels"}

words_dict = {
    "Long_words": ["cooperate", "independent"],
    "Short_Long_words": ["cooperate", "in"],
    "Short_words": ["out", "in", "up"],
    "Vowels": ["a", "i", "u"]
}

numeric_labels = {
    "Long_words": {"cooperate": 0, "independent": 1},
    "Short_Long_words": {"cooperate": 0, "in": 1},
    "Short_words": {"out": 0, "in": 1, "up": 2},
    "Vowels": {"a": 0, "i": 1, "u": 2}

}


# retrieves the MPC(Mean Phase Coherance) feature matrix for given EEG 64 channel
def MPC(eeg):
    channels = eeg.shape[0]
    mpc_matrix = np.zeros((channels, channels), dtype=float)

    def MPC_feature(i, j):
        signal_a = np.unwrap(np.angle(scipy.signal.hilbert(eeg[i])))
        signal_b = np.unwrap(np.angle(scipy.signal.hilbert(eeg[j])))
        phase_diff = np.exp((signal_a - signal_b) * 1j)
        return np.absolute(np.mean(phase_diff))

    for i in range(channels):
        for j in range(channels):
            if i <= j:
                mpc_matrix[i, j] = MPC_feature(i, j)
            else:
                mpc_matrix[i, j] = mpc_matrix[j, i]
    return mpc_matrix


# retrieves the MSC(Magnitude Phase Coherance) feature matrix for given EEG 64 channel
def MSC(eeg):
    channels = eeg.shape[0]
    msc_matrix = np.zeros((channels, channels), dtype=float)

    for i in range(channels):
        for j in range(channels):
            if i <= j:
                msc_matrix[i, j] = np.mean(scipy.signal.coherence(
                    eeg[i], eeg[j], window=scipy.signal.windows.hamming(32), fs=256)[1])
            else:
                msc_matrix[i, j] = msc_matrix[j, i]
    return msc_matrix


# alpha beta gamma filtering for every eeg electrode
def alpha_beta_gamma_extractor(eeg):
    a = scipy.signal.butter(8, [8, 13], 'bandpass', fs=256, output='sos')
    b = scipy.signal.butter(8, [13, 30], 'bandpass', fs=256, output='sos')
    g = scipy.signal.butter(8, [30, 70], 'bandpass', fs=256, output='sos')

    alpha = np.zeros_like(eeg)
    beta = np.zeros_like(eeg)
    gamma = np.zeros_like(eeg)

    for i in range(eeg.shape[0]):
        alpha[i] = scipy.signal.sosfilt(a, eeg[i])
        beta[i] = scipy.signal.sosfilt(b, eeg[i])
        gamma[i] = scipy.signal.sosfilt(g, eeg[i])

    return [alpha, beta, gamma]


# reutrn Image form of the eeg from alpha beta gamma bands and MPC and MSC feature matrix
def EEG_Image(eeg, **kwargs):
    eeg_channles = alpha_beta_gamma_extractor(eeg)
    Image = np.zeros((eeg.shape[0], eeg.shape[0], 3), dtype=float)
    for i in range(3):
        eeg_mpc = MPC(eeg_channles[i])
        eeg_msc = MPC(eeg_channles[i])
        n = eeg_mpc.shape[0]
        for p in range(n):
            for q in range(n):
                if p < q:
                    Image[p, q, i] = eeg_mpc[p, q]
                elif p > q:
                    Image[p, q, i] = eeg_msc[p, q]
    return Image
