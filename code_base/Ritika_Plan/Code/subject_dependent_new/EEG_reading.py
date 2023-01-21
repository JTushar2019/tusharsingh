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
            if decided_channels.issubset(edf.info.ch_names):
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


def modify_and_store_EEG(X, Y, augment, pathology_distribution):
    store_path = temp_folder_path
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)
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


def data_augment(x, y, augmentation, store_path):
    global window_size

    window_size = int(time_window * sampling_frequency)
    stride = window_size
    
    eeg = mne.io.read_raw(x, preload=True, verbose=False)
    eeg = eeg.pick_channels(decided_channels)
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

def over_sampling(X_train, Y_train):

    l, f = np.unique(Y_train, return_counts = True)

    avarage = int(np.mean(f))

    min_class = l[np.argmin(f)]
    max_class = l[np.argmax(f)]

    difference = avarage - np.min(f)



    index = (Y_train == min_class)
    X_train_min_class = X_train[index]
    Y_train_min_class = Y_train[index]

    X_train_max_class = X_train[~index]
    Y_train_max_class = Y_train[~index]

    random_indices = np.random.choice(X_train_max_class.shape[0], size=avarage, replace=False)

    X_train_max_class = X_train_max_class[random_indices]
    Y_train_max_class = Y_train_max_class[random_indices]

    augmented_X_train = []
    augmented_Y_train = []
    name = 'augmented_min_X_'
    for i in range(difference + (int(avarage * 0.6))):
        a,b = [np.load(x) for x in np.random.choice(X_train_min_class, size=2, replace=True)]
        new = np.concatenate([a,b], axis = 1)
        start = a.shape[1] // 2
        end = start + time_window * sampling_frequency
        new = new[:, start: end]

        this_name = f'{temp_folder_path}/{name}_{i}.npy'
        with open(this_name, 'wb') as f:
            np.save(f, new)
        augmented_X_train.append(this_name)
        augmented_Y_train.append(min_class)

    augmented_X_train = np.array(augmented_X_train)
    augmented_Y_train = np.array(augmented_Y_train)

    if augmented_X_train.shape[0] > 0:
        X_train_min_class = np.concatenate([X_train_min_class, augmented_X_train])
        Y_train_min_class = np.concatenate([Y_train_min_class, augmented_Y_train])


    augmented_X_train = []
    augmented_Y_train = []
    name = 'augmented_max_X_'
    for i in range(int(avarage * 0.6)):
        a,b = [np.load(x) for x in np.random.choice(X_train_max_class, size=2, replace=True)]
        new = np.concatenate([a,b], axis = 1)
        start = a.shape[1] // 2
        end = start + time_window * sampling_frequency
        new = new[:, start: end]

        this_name = f'{temp_folder_path}/{name}_{i}.npy'
        with open(this_name, 'wb') as f:
            np.save(f, new)
        augmented_X_train.append(this_name)
        augmented_Y_train.append(max_class)

    augmented_X_train = np.array(augmented_X_train)
    augmented_Y_train = np.array(augmented_Y_train)

    if augmented_X_train.shape[0] > 0:
        X_train_max_class = np.concatenate([X_train_max_class, augmented_X_train])
        Y_train_max_class = np.concatenate([Y_train_max_class, augmented_Y_train])
    
    X_train = np.concatenate([X_train_max_class, X_train_min_class])
    Y_train = np.concatenate([Y_train_max_class, Y_train_min_class])


    random_indices = np.random.permutation(Y_train.shape[0])

    print(f'now_total_train 30sec samples -    {Y_train.shape[0]}')
    print(f'train_augmeted samples  -    {difference}   {100 * (difference + (avarage // 3)) // X_train.shape[0]:0.2f}% of NEW size')
    return X_train[random_indices], Y_train[random_indices]


def preprocess_whole_data():

    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    os.mkdir(temp_folder_path)


    print("pre-processing data...")
    X, Y = complete_data_path()
    print(f'\n{list(Counter(Y))}\n')
    print(f'observed min sampling_rate {sampling_frequency}')
    print(
        f'observed max highpass = {highpass:0.2f} \nobserved min lowpass = {lowpass:0.2f}\n')

    pathology_distribution = defaultdict(int)
    X, Y = [np.array(each) for each in modify_and_store_EEG(X, Y, False, pathology_distribution)]

    print(f'original_pathology_epoc_distribution in %')
    for each in pathology_distribution:
        pathology_distribution[each] = 100 * (pathology_distribution[each] / X.shape[0])    
        print(f"\t{each:35s} : {pathology_distribution[each]:0.2f}%")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = split_ratio, random_state = 123, stratify = Y)
    
    with open(f'{working_data_path}/X_test.npy', 'wb') as f:
        np.save(f, X_test)
    with open(f'{working_data_path}/Y_test.npy', 'wb') as f:
        np.save(f, Y_test)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size = 0.9, random_state = 123, stratify = Y_train)

    with open(f'{working_data_path}/X_val.npy', 'wb') as f:
        np.save(f, X_val)
    with open(f'{working_data_path}/Y_val.npy', 'wb') as f:
        np.save(f, Y_val)

    for name, size in zip(['train_size', 'val_size', 'test_size'],[Y_train.shape[0], Y_val.shape[0], Y_test.shape[0]]):
        print(f'\t{name:15s}: {size}')
    print()

    if train_over_sampling:
        X_train, Y_train = over_sampling(X_train, Y_train)

    temp = np.load(X_train[0])
    print(f'single sample dimention = {temp.shape}')

    with open(f'{working_data_path}/X_train.npy', 'wb') as f:
        np.save(f, X_train)
    with open(f'{working_data_path}/Y_train.npy', 'wb') as f:
        np.save(f, Y_train)

    print(f'X and Y are stored at \n{working_data_path}')
    

import os, sys
from datetime import datetime

def preprocess_and_record():
    logging_file = open(tracking_file_path, "a")
    temp = sys.stdout
    sys.stdout = logging_file

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y %H:%M:%S\n\n")


    print("date and time =", dt_string)
    print(f'picked channels = {decided_channels}')

    preprocess_whole_data()
    sys.stdout = temp
    logging_file.close()


if __name__ == '__main__':
    preprocess_and_record()
