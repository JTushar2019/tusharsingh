import scipy.io as sio, numpy as np, scipy
import os, re, random,concurrent.futures

folder_path = {"Long_words": "/home/tusharsingh/DATAs/speech_EEG/Long_words",
        "Short_Long_words": "/home/tusharsingh/DATAs/speech_EEG/Short_Long_words",
        "Short_words": "/home/tusharsingh/DATAs/speech_EEG/Short_words",
        "Vowels": "/home/tusharsingh/DATAs/speech_EEG/Vowels"}

Labels = {"Long_words": {1: "cooperate",2: "independent"},
        "Short_Long_words": {1:"cooperate",2:"in"},
        "Short_words": {1:"out",2:"in",3:"up"},
        "Vowels": {1:"a",2:"i",3:"u"}}

eeg_image_folder_path = {"Long_words": "/home/tusharsingh/DATAs/speech_EEG/EEG_image/Long_words",
        "Short_Long_words": "/home/tusharsingh/DATAs/speech_EEG/EEG_image/Short_Long_words",
        "Short_words": "/home/tusharsingh/DATAs/speech_EEG/EEG_image/Short_words",
        "Vowels": "/home/tusharsingh/DATAs/speech_EEG/EEG_image/Vowels"}


# loads EEGs from the path given 
def load_EEG(path, words):
    eeg = []
    labels = []
    patient_id = []
    name = []
    for subject in os.scandir(path):
        if subject.is_file() and subject.name.endswith('.mat'):
            mat = sio.loadmat(subject.path)['eeg_data_wrt_task_rep_no_eog_256Hz_last_beep']
            for label in range(mat.shape[0]):
                for j in range(mat[label].shape[0]):
                    eeg.append(mat[label][j][:64,:])
                    labels.append(words[label + 1])
                    patient_id.append(int(re.search("[0-9]+", subject.name).group(0)))
                    name.append(f"patient_{patient_id[-1]}_word_{labels[-1]}_trial_{j+1}")
    return [eeg,labels,patient_id, name]

# return augmented_data with given window_size and stride
def augmented_data(path, words, window_size = 256, stride = 64):
    EEG, Labels, Patient_id, file_name = load_EEG(path, words)
    X = []
    Y = []
    id = []
    augmented_file_name = []
    for eeg, label, patient_id, name in zip(EEG, Labels, Patient_id, file_name):
        for start in range(0, eeg.shape[1] - window_size + 1, stride):
            X.append(eeg[:,start: start + window_size])
            Y.append(label)
            id.append(patient_id)
            augmented_file_name.append(f"{name}_{start//stride + 1}")

    return [X, Y, id, augmented_file_name]


# retrieves the MPC(Mean Phase Coherance) feature matrix for given EEG 64 channel
def MPC(eeg):
    channels = eeg.shape[0]
    mpc_matrix = np.zeros((channels, channels), dtype = float)

    def MPC_feature(i,j):
        signal_a = np.unwrap(np.angle(scipy.signal.hilbert(eeg[i])))
        signal_b = np.unwrap(np.angle(scipy.signal.hilbert(eeg[j])))
        phase_diff = np.exp((signal_a - signal_b) * 1j)
        return np.absolute(np.mean(phase_diff))
        
    for i in range(channels):
        for j in range(channels):
            if i <= j:
                mpc_matrix[i, j] = MPC_feature(i,j)
            else:
                mpc_matrix[i, j] = mpc_matrix[j, i]
    return mpc_matrix

    
# retrieves the MSC(Magnitude Phase Coherance) feature matrix for given EEG 64 channel
def MSC(eeg):
    channels = eeg.shape[0]
    msc_matrix = np.zeros((channels, channels), dtype = float)
        
    for i in range(channels):
        for j in range(channels):
            if i <= j:
                msc_matrix[i, j] = np.mean(scipy.signal.coherence(eeg[i], eeg[j], window = scipy.signal.windows.hamming(32), fs = 256)[1])
            else:
                msc_matrix[i, j] = msc_matrix[j, i]
    return msc_matrix


# alpha beta gamma filtering for every eeg electrode    
def alpha_beta_gamma_extractor(eeg):
    a = scipy.signal.butter(8, [8,13], 'bandpass', fs=256, output='sos')
    b = scipy.signal.butter(8, [13,30], 'bandpass', fs=256, output='sos')
    g = scipy.signal.butter(8, [30,70], 'bandpass', fs=256, output='sos')

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
    Image = np.zeros((eeg.shape[0],eeg.shape[0],3), dtype=float)
    for i in range(3):
        eeg_mpc = MPC(eeg_channles[i])
        eeg_msc = MPC(eeg_channles[i])
        n = eeg_mpc.shape[0]
        for p in range(n):
            for q in range(n):
                if p < q:
                    Image[p,q,i] = eeg_mpc[p,q]
                elif p > q:
                    Image[p,q,i] = eeg_msc[p,q]
    return Image


# it extracts eeg Image and writes it into given location with proper name
def eeg_image_folder_maker(store_path, data, start, stop):
    X, Y, id, file_name = data
    for i in range(start,stop):
        with open(f"{store_path}/{file_name[i]}", 'wb') as f:
            np.save(f, EEG_Image(X[i]))
            np.save(f, Y[i])
            np.save(f, id[i])   


def parllel_feature_extracting(folder_name):
    data = augmented_data(folder_path[folder_name], Labels[folder_name], 256, 64)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        N = len(data[1])
        cores = 40
        per_core = N // cores
        for i in range(0, N, per_core):
            print(".", end="")
            executor.submit(eeg_image_folder_maker, eeg_image_folder_path[folder_name], data, i, min(N,i + per_core))
    print("\nSuccesfully Done -> ", len(os.listdir(eeg_image_folder_path[folder_name])) == N)

if __name__ == "__main__":
    # parllel_feature_extracting('Long_words')
    # parllel_feature_extracting('Short_Long_words')
    # parllel_feature_extracting('Short_words')
    # parllel_feature_extracting('Vowels')