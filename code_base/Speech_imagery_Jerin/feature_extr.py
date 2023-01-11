import numpy as np
import scipy
from global_variables import *

# retrieves the MPC(Mean Phase Coherance) feature matrix for given EEG 64 channel
def MPC(eeg):
    channels = eeg.shape[0]
    mpc_matrix = np.zeros((channels, channels), dtype=float)

    def MPC_feature(i, j):
        signal_a = np.unwrap(np.angle(scipy.signal.hilbert(eeg[i])))
        signal_b = np.unwrap(np.angle(scipy.signal.hilbert(eeg[j])))
        phase_diff = np.absolute(np.exp((signal_a - signal_b) * 1j))
        return np.mean(phase_diff)

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
    msc_matrix = np.zeros((channels, channels, 3), dtype=float)
    for i in range(channels):
        for j in range(channels):
            if i <= j:
                temp = scipy.signal.coherence(
                    eeg[i], eeg[j], window = scipy.signal.windows.hamming(51) , nfft = 256, fs=256)
                t1 = (temp[0] <= 8).astype(bool)
                t2 = (temp[0] <= 13).astype(bool)
                t3 = (temp[0] <= 30).astype(bool)
                t4 = (temp[0] <= 70).astype(bool)
                alpha = np.mean(temp[1][~t1 & t2])
                beta = np.mean(temp[1][~t2 & t3])
                gamma = np.mean(temp[1][~t3 & t4])
                msc_matrix[i,j,0] = alpha
                msc_matrix[i,j,1] = beta
                msc_matrix[i,j,2] = gamma
            else:
                msc_matrix[i, j, 0] = msc_matrix[j, i, 0]
                msc_matrix[i, j, 1] = msc_matrix[j, i, 1]
                msc_matrix[i, j, 2] = msc_matrix[j, i, 2]
    return msc_matrix


# alpha beta gamma filtering for every eeg electrode
def alpha_beta_gamma_extractor(eeg):
    a = scipy.signal.butter(1, [8, 13], 'bandpass', fs=256, output='sos')
    b = scipy.signal.butter(1, [13, 30], 'bandpass', fs=256, output='sos')
    g = scipy.signal.butter(1, [30, 70], 'bandpass', fs=256, output='sos')

    alpha = scipy.signal.sosfilt(a, eeg, axis = 1)
    beta = scipy.signal.sosfilt(b, eeg, axis = 1)
    gamma = scipy.signal.sosfilt(g, eeg, axis = 1)

    return [alpha, beta, gamma]


# reutrn Image form of the eeg from alpha beta gamma bands and MPC and MSC feature matrix
def EEG_Image(eeg, **kwargs):
    eeg_channles = alpha_beta_gamma_extractor(eeg)
    Image = MSC(eeg)
    for i in range(3):
        eeg_mpc = MPC(eeg_channles[i])
        n = eeg_mpc.shape[0]
        for p in range(n):
            Image[p,p,i] = 0
            for q in range(p + 1, n):
                Image[p, q, i] = eeg_mpc[p, q]
    return Image


if __name__ == "__main__":
    from pre_processing import load_EEG
    eeg_path, label = load_EEG("Long_words", 2)
    eeg = eeg_path[0]
    with open(eeg, 'rb') as f:
        eeg = np.load(f)
    alpha, beta, gamma = alpha_beta_gamma_extractor(eeg)

    print(EEG_Image(eeg)[:,:,0])
