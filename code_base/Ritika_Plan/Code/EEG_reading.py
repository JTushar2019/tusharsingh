from collections import Counter, defaultdict
import shutil
from global_variables import *
import scipy.io as sio
import numpy as np
import os
import mne
import re
import concurrent.futures


def get_file_path_for(pathology):
    global sampling_frequency, highpass, lowpass
    X = []
    Y = []
    for subject_file in os.scandir(data_folder_path):
        if subject_file.name.endswith('edf') and re.search(f"{pathology}[0-9]+", subject_file.name) is not None:
            # print(re.search(f"{pathology}[0-9]+", subject_file.name).group(0), pathology)
            edf = mne.io.read_raw(subject_file, verbose=0)
            temp = [x.replace("-", "") for x in edf.ch_names]
            if set(decided_channels).issubset(temp):
                X.append(f'{data_folder_path}/{subject_file.name}')
                Y.append(pathology_dict[pathology])
                sampling_frequency = min(sampling_frequency, edf.info['sfreq'])
                lowpass = min(lowpass, edf.info['lowpass'])
                highpass = max(highpass, edf.info['highpass'])

    return X, Y


def complete_data_path():
    X = []
    Y = []
    for each in pathology_dict.keys():
        x, y = get_file_path_for(each)
        X.extend(x)
        Y.extend(y)
    return X, Y


def modify_and_store_EEG(X, Y, pathology_distribution):
    store_path = temp_folder_path
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)
    new_X, new_Y = [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(max(1, os.cpu_count()*0.80))) as executor:
        futures = []
        for x, y in zip(X, Y):
            futures.append(executor.submit(
                data_augment, x, y, store_path))
        for future in concurrent.futures.as_completed(futures):
            x = future.result()[0]
            y = future.result()[1]
            pathology_distribution[y[0]] += len(x)
            new_X.extend(x)
            new_Y.extend(y)

    return new_X, new_Y


def data_augment(x, y, store_path):
    global window_size, stride
    window_size = int(time_window * sampling_frequency)
    stride = int(time_overlap * sampling_frequency)
    eeg = mne.io.read_raw(x, preload=True, verbose=False)
    eeg = eeg.pick_channels(dicided_channels_name)
    eeg = eeg.filter(highpass, lowpass, verbose=False)
    eeg = eeg.resample(sampling_frequency)
    eeg = eeg.get_data()

    eeg_file_name = x.split('/')[-1].removesuffix(".edf")
    X = []
    for start in range(0, eeg.shape[1], stride):
        end = start + window_size
        if end >= eeg.shape[1]:
            continue
        temp = eeg[:, start:end]
        name = f'{store_path}/{eeg_file_name}_{start // window_size + 1}.npy'
        with open(name, 'wb') as f:
            np.save(f, temp)
        X.append(name)
    Y = [y]*len(X)
    return X, Y


def preprocess_whole_data():
    print("pre-processing data...")
    X, Y = complete_data_path()
    print(Counter(Y))
    print(f'observed min sampling_rate {sampling_frequency}')
    print(
        f'observed max highpass = {highpass} \nobserved min lowpass = {lowpass}')

    pathology_distribution = defaultdict(int)
    X, Y = modify_and_store_EEG(X, Y, pathology_distribution)
    X = np.array(X)
    Y = np.array(Y)

    print(f'total 30sec samples - {X.shape[0]}')
    temp = np.load(X[0])
    print(f'single sample dimention = {temp.shape}')
    print(f'pathology_distribution in %')
    for each in pathology_distribution:
        pathology_distribution[each] = 100 * (pathology_distribution[each] / X.shape[0])    
        print(f"{each} : {pathology_distribution[each]:0.2f}%")


    working_data_path = f'{data_folder_path}/..'
    with open(f'{working_data_path}/X.npy', 'wb') as f:
        np.save(f, X)
    with open(f'{working_data_path}/Y.npy', 'wb') as f:
        np.save(f, Y)

    print(f'X.npy, Y.npy are stored at \n{working_data_path}')
    return X, Y


if __name__ == '__main__':
    preprocess_whole_data()
