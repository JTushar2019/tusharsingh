from collections import Counter, defaultdict
import shutil
from global_variables import *
import numpy as np
import os
import mne
import re
import concurrent.futures
from sklearn.model_selection import train_test_split


def get_file_path_for(pathology):
    global sampling_frequency, highpass, lowpass
    X = []
    Y = []
    for subject_file in os.scandir(data_folder_path):
        if subject_file.name.endswith('fif') and re.search(f"{pathology}[0-9]+", subject_file.name) is not None:
            name = f'{data_folder_path}/{subject_file.name}'
            edf = mne.io.read_raw(name, verbose=0)

            if set(decided_channels).issubset(edf.info.ch_names):
                X.append(f'{data_folder_path}/{subject_file.name}')
                Y.append(pathology_dict[pathology])
                sampling_frequency = min(sampling_frequency, edf.info['sfreq'])
                lowpass = min(lowpass, edf.info['lowpass'])
                highpass = max(highpass, edf.info['highpass'])

    return X, Y


def complete_data_path():
    X = []
    Y = []
    :for each in pathology_dict.keys()
        x, y = get_file_path_for(each)
        X.extend(x)
        Y.extend(y)

    print(f'Total subject distribution \n\t{Counter(Y)}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, train_size=split_ratio, random_state=123, stratify=Y)

    print(f'Train distribution \n\t{Counter(y_train)}')
    print(f'Test distribution  \n\t{Counter(y_test)}')

    print(
        f"\ntraining on {str([x.split('/')[-1].removesuffix('_raw.fif') for x in X_train])}")
    print(
        f"\ntesting on {str([x.split('/')[-1].removesuffix('_raw.fif') for x in X_test])}\n")

    return X_train, X_test, y_train, y_test


def modify_and_store_EEG(X, Y, pathology_distribution, augment=False):
    store_path = temp_folder_path

    new_X, new_Y = [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(max(1, os.cpu_count()*0.80))) as executor:
        futures = []
        for x, y in zip(X, Y):
            futures.append(executor.submit(
                data_augment, x, y, augment, store_path))
        for future in concurrent.futures.as_completed(futures):
            x = future.result()[0]
            y = future.result()[1]
            pathology_distribution[y[0]] += len(x)
            new_X.extend(x)
            new_Y.extend(y)

    return new_X, new_Y


def data_augment(x, y, augment, store_path):
    global window_size, pathology_time_overlap

    time_overlap = pathology_time_overlap[y]
    if not augment:
        time_overlap = False

    window_size = int(time_window * sampling_frequency)
    stride = window_size - int(time_overlap * sampling_frequency)
    eeg = mne.io.read_raw(x, preload=True, verbose=False)
    eeg = eeg.pick_channels(dicided_channels_name)
    eeg = eeg.filter(highpass, lowpass, verbose=False)
    eeg = eeg.resample(sampling_frequency)
    eeg = eeg.get_data()

    eeg_file_name = x.split('/')[-1].removesuffix("_raw.fif")
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

    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    os.mkdir(temp_folder_path)

    X_train, X_test, y_train, y_test = complete_data_path()
    print(f'observed min sampling_rate {sampling_frequency}')
    print(
        f'observed max highpass = {highpass:0.2f} \nobserved min lowpass = {lowpass:0.2f}\n')

    train_pathology_distribution = defaultdict(int)
    X_train, y_train = modify_and_store_EEG(
        X_train, y_train, train_pathology_distribution, augment=True)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    test_pathology_distribution = defaultdict(int)
    X_test, y_test = modify_and_store_EEG(
        X_test, y_test, test_pathology_distribution, augment=False)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(
        f'total 30sec samples for Train: {X_train.shape[0]}       Test: {X_test.shape[0]}')

    temp = np.load(X_train[0])

    print(f'single sample dimention = {temp.shape}')

    print(f'\npathology_distribution in Train and Test in %\n')

    for each in train_pathology_distribution:
        train_pathology_distribution[each] = 100 * \
            (train_pathology_distribution[each] / X_train.shape[0])
        test_pathology_distribution[each] = 100 * \
            (test_pathology_distribution[each] / X_test.shape[0])

        print(
            f"{each} :\n\t\tTrain: {train_pathology_distribution[each]:0.2f}%  over_sampling: {pathology_time_overlap[each]} Sec  Test : {test_pathology_distribution[each]:0.2f}%")

    working_data_path = f'{data_folder_path}/..'
    with open(f'{working_data_path}/X_train.npy', 'wb') as f:
        np.save(f, X_train)
    with open(f'{working_data_path}/Y_train.npy', 'wb') as f:
        np.save(f, y_train)

    with open(f'{working_data_path}/X_test.npy', 'wb') as f:
        np.save(f, X_test)
    with open(f'{working_data_path}/Y_test.npy', 'wb') as f:
        np.save(f, y_test)

    print(f'\nTrain and Test list are stored at \n{working_data_path}')


if __name__ == '__main__':
    preprocess_whole_data()
