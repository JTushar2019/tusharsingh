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


def modify_and_store_EEG(X, Y, time_window=30):
    store_path = temp_folder_path
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)
    new_X, new_Y = [], []

    window_size = time_window * sampling_frequency
    print(window_size)

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(max(1, os.cpu_count()*0.80))) as executor:
        futures = []
        for x, y in zip(X, Y):
            futures.append(executor.submit(
                data_augment, x, y, window_size, store_path))
        for future in concurrent.futures.as_completed(futures):
            new_X.extend(future.result()[0])
            new_Y.extend(future.result()[1])

    return new_X, new_Y


def data_augment(x, y, window_size, store_path):
    eeg = mne.io.read_raw(x, preload=True, verbose=False)
    eeg = eeg.pick_channels(dicided_channels_name)
    eeg = eeg.filter(highpass, lowpass, verbose=False)
    eeg = eeg.resample(sampling_frequency)
    eeg = eeg.get_data()

    eeg_file_name = x.split('/')[-1].removesuffix(".edf")
    X = []
    Y = []
    for start in range(0, eeg.shape[1], window_size):
        end = start + window_size
        if end >= eeg.shape[1]:
            continue
        temp = eeg[:, start:end]
        name = f'{store_path}/{eeg_file_name}_{start // window_size + 1}.npy'
        with open(name, 'wb') as f:
            np.save(f, temp)
        X.append(name)
        Y.append(y)

    return X, Y


if __name__ == '__main__':
    from collections import Counter
    X, Y = complete_data_path()
    print(Counter(Y))
    print(len(X), len(Y))
    print(f'min sampling_rate {sampling_frequency}')
    print(f'max highpass = {highpass} \nmin lowpass = {lowpass}')
    # print(X[:2])

    X, Y = modify_and_store_EEG(X, Y, 30)
    X = np.array(X)
    Y = np.array(Y)
    working_data_path = "/home/tusharsingh/code_base/Ritika_Plan/Data"
    with open(f'{working_data_path}/X.npy', 'wb') as f:
        np.save(f, X)
    with open(f'{working_data_path}/Y.npy', 'wb') as f:
        np.save(f, Y)

    print(X[:2])
    print(Y[:2])
    print(X.shape, Y.shape)


{'Nocturnal frontal lobe epilepsy': 38, 'REM behavior disorder': 22, 'Periodic leg movements': 10,
    'Insomnia': 9, 'controls': 6, 'Narcolepsy': 5, 'Sleep-disordered breathing': 4, 'Bruxism': 2}