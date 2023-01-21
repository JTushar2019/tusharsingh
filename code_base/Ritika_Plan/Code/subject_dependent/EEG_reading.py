from collections import Counter, defaultdict
import shutil
from global_variables import *
import numpy as np
import os, pickle
import mne
import re
import concurrent.futures
from sklearn.model_selection import train_test_split


sampling_frequency = 256
highpass = 0.3
lowpass = 30


def get_file_path_for(pathology, decided_channels, pathology_dict):
    global sampling_frequency, highpass, lowpass
    X = []
    Y = []
    for subject_file in os.scandir(data_folder_path):
        if subject_file.name.endswith('fif') and re.search(f"{pathology}[0-9]+", subject_file.name) is not None:
            name = f'{data_folder_path}/{subject_file.name}'
            edf = mne.io.read_raw(name, verbose=0)
            if set(decided_channels).issubset(set(edf.info.ch_names)):
                X.append(f'{data_folder_path}/{subject_file.name}')
                Y.append(pathology_dict[pathology])
                sampling_frequency = min(sampling_frequency, edf.info['sfreq'])
                lowpass = min(lowpass, edf.info['lowpass'])
                highpass = max(highpass, edf.info['highpass'])
    return X, Y


def complete_data_path(decided_channels, pathology_dict):
    X = []
    Y = []
    for each in pathology_dict.keys():
        x, y = get_file_path_for(each, decided_channels, pathology_dict)
        X.extend(x)
        Y.extend(y)
    return X, Y


def modify_and_store_EEG(decided_channels, X, Y, pathology_distribution):
    store_path = temp_folder_path
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)
    new_X, new_Y = [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(max(1, os.cpu_count()*0.80))) as executor:
        futures = []
        for x, y in zip(X, Y):
            futures.append(executor.submit(
                data_augment, decided_channels, x, y, store_path))
        for future in concurrent.futures.as_completed(futures):
            x = future.result()[0]
            y = future.result()[1]
            pathology_distribution[y[0]] += len(x)
            new_X.extend(x)
            new_Y.extend(y)

    # for x,y in zip(X,Y):
    #     x, y = data_augment(decided_channels, x, y, store_path)
    #     pathology_distribution[y[0]] += len(x)
    #     new_X.extend(x)
    #     new_Y.extend(y)


    return new_X, new_Y


def data_augment(decided_channels, x, y, store_path):
    global window_size

    window_size = int(time_window * sampling_frequency)
    eeg = mne.io.read_raw(x, preload=True, verbose=False)
    # print(eeg.info.ch_names)
    eeg = eeg.filter(highpass, lowpass, verbose=False, picks = decided_channels)
    eeg = eeg.pick_channels(decided_channels, ordered = True)
    eeg = eeg.resample(sampling_frequency)
    eeg = eeg.get_data()

    eeg_file_name = x.split('/')[-1].removesuffix("_raw.fif")
    X = []
    for start in range(0, eeg.shape[1], window_size):
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


def preprocess_whole_data(decided_channels, pathology_dict):

    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    os.mkdir(temp_folder_path)


    print("pre-processing data...")
    X, Y = complete_data_path(decided_channels, pathology_dict)
    print(f'\n{list(Counter(Y))}\n')
    print(f'observed min sampling_rate {sampling_frequency}')
    print(
        f'observed max highpass = {highpass:0.2f} \nobserved min lowpass = {lowpass:0.2f}\n')

    pathology_distribution = defaultdict(int)
    X, Y = [np.array(each) for each in modify_and_store_EEG(decided_channels, X, Y, pathology_distribution)]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.reshape(-1,1), train_size = split_ratio, random_state = 123, stratify = Y.reshape(-1,1))

    working_data_path = f'{data_folder_path}/..'
    with open(f'{working_data_path}/X_test.npy', 'wb') as f:
        np.save(f, X_test)
    with open(f'{working_data_path}/Y_test.npy', 'wb') as f:
        np.save(f, Y_test)
    

    augmented_X_train = []
    augmented_Y_train = []
    Y_train = Y_train.flatten()

    l, f = np.unique(Y_train, return_counts = True)
    min_class = l[np.argmin(f)]
    difference = np.max(f) - np.min(f)

    X_train_min_class = X_train[Y_train == min_class]

    name = 'augmented_X_'
    for i in range(difference):
        a,b = [np.load(x) for x in np.random.choice(X_train_min_class, size=2, replace=True)]
        new = np.concatenate([a,b], axis = 1)
        start = a.shape[1] // 2
        end = int(start + time_window * sampling_frequency)
        new = new[:, start: end]

        this_name = f'{temp_folder_path}/{name}_{i}.npy'
        with open(this_name, 'wb') as f:
            np.save(f, new)
        augmented_X_train.append(this_name)
        augmented_Y_train.append(min_class)
    
    augmented_X_train = np.array(augmented_X_train)
    augmented_Y_train = np.array(augmented_Y_train)

    if augmented_Y_train.shape[0] > 0:
        Y_train = np.concatenate([Y_train, augmented_Y_train])
        X_train = np.concatenate([X_train, augmented_X_train])


    print(f'total 30sec samples -    {Y_train.shape[0]}')
    print(f'total_augmeted samples - {augmented_Y_train.shape[0]}   {(100 * augmented_Y_train.shape[0]) // X_train.shape[0]:0.2f}%')

    temp = np.load(X[0])
    print(f'single sample dimention = {temp.shape}')
    
    print(f'original_pathology_epoc_distribution in %')
    for each in pathology_distribution:
        pathology_distribution[each] = 100 * (pathology_distribution[each] / X.shape[0])    
        print(f"\t{each:35s} : {pathology_distribution[each]:0.2f}%")

    working_data_path = f'{data_folder_path}/..'
    with open(f'{working_data_path}/X_train.npy', 'wb') as f:
        np.save(f, X_train)
    with open(f'{working_data_path}/Y_train.npy', 'wb') as f:
        np.save(f, Y_train)

    print(f'X and Y are stored at \n{working_data_path}')

    with open(f'{saved_model_path}/sampling_frequency.pickle', 'wb') as handle:
        pickle.dump(sampling_frequency, handle)
    

if __name__ == '__main__':
    pass



