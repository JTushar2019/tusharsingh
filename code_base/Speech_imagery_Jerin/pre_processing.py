from feature_extr import *
import scipy.io as sio
import numpy as np
import scipy
import os
import shutil
import re
import random
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader

matrix_to_load = "eeg_data_wrt_task_rep_no_eog_256Hz_last_beep"

def load_EEG(type, subject_no):
    path = folder_path[type]
    words = words_dict[type]
    for subject_file in os.scandir(path):
        if not (subject_file.is_file() and subject_file.name.endswith('.mat') and
                int(re.search("[0-9]+", subject_file.name).group(0)) == subject_no):
            continue
        mat = sio.loadmat(subject_file.path)[matrix_to_load]
        
        temp = f"{path}/temp_files"
        if not os.path.exists(temp):
            os.mkdir(temp)
        temp = f"{temp}/{subject_no}"

        if not os.path.exists(temp):
            os.mkdir(temp)
        X = []
        for index, eeg in np.ndenumerate(mat):
            temp2 = f"{temp}/{words[index[0]]}_{index[1] + 1}.npy"
            X.append(temp2)
            if not os.path.exists(temp2):
                np.save(temp2, eeg)
    return X


def get_train_preprocessed_data(X_test, pre_process, transformer):
    window_size = 256
    stride = 64
    new_X = []

    with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 20) as executor:
        futures = []
        for i in range(len(X_test)):
            futures.append(executor.submit(
                pre_process, X_test[i], transformer, window_size, stride))
        for future in concurrent.futures.as_completed(futures):
            new_X.extend(future.result())

    return new_X, [each.split("_")[0] for each in new_X]


def get_test_preprocessed_data(X_test, pre_process, transformer):
    window_size = 256
    stride = 64
    new_X = []

    with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 20) as executor:
        futures = []
        for i in range(len(X_test)):
            futures.append(executor.submit(
                pre_process, X_test[i], transformer, window_size, stride))
        for future in concurrent.futures.as_completed(futures):
            new_X.extend(future.result())

    return new_X, [each.split("_")[0] for each in new_X]


def train_pre_process(X, transformer,  window_size, stride):
    new_X = []
    with open(X, 'rb') as f:
        eeg = np.load(f)
    temp = X.removesuffix(".npy")
    for start in range(0, eeg.shape[1] - window_size + 1, stride):
        new_X.append(f"{temp}_{start//stride + 1}.npy")
        if os.path.exists(new_X[-1]): continue
        with open(new_X[-1], "wb") as f:
            np.save(f, transformer(eeg[:, start:start + window_size]))
    return new_X


def test_pre_process(X, transformer,  window_size, stride):
    with open(X, 'rb') as f:
        eeg = np.load(f)
    temp = X.removesuffix(".npy")
    new_X = f"{temp}_test.npy"
    if not os.path.exists(new_X):
        np.save(new_X, transformer(eeg))
    return new_X


if __name__ == "__main__":

    type = "Long_words"
    
    for each in [2,3,6,7,9,11]:
        temp = load_EEG(type, each)
        X = get_train_preprocessed_data(temp, train_pre_process, EEG_Image)
        print(type, len(X[0]),len(X[1]))
        
    
    # type = "Short_Long_words"

    # for each in [1,5,8,9,10,14]:
    #     temp = load_EEG(type, each)
    #     X = get_train_preprocessed_data(temp, train_pre_process, EEG_Image)
    #     print(type, len(X[0]),len(X[1]))
    

    # type = "Short_words"

    # for each in [1,3,5,6,8,12]:
    #     temp = load_EEG(type, each)
    #     X = get_train_preprocessed_data(temp, train_pre_process, EEG_Image)
    #     print(type, len(X[0]),len(X[1]))
    


    # type = "Vowels"

    # for each in [4,5,8,9,11,12,13,15]:
    #     temp = load_EEG(type, each)
    #     X = get_preprocessed_data(temp, train_pre_process, EEG_Image)
    #     print(type, len(X[0]),len(X[1]))
    
