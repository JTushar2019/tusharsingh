import numpy as np
import os
from collections import Counter
from EEG_reading import preprocess_whole_data
from global_variables import *
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


def read_and_oneHotencode():
    if not os.path.exists(f'{working_data_path}/X_train.npy'):
        preprocess_whole_data()
    else:
        with open(f'{working_data_path}/X_train.npy', 'rb') as f:
            X_train = np.load(f)
        with open(f'{working_data_path}/Y_train.npy', 'rb') as f:
            y_train = np.load(f)
        with open(f'{working_data_path}/X_test.npy', 'rb') as f:
            X_test = np.load(f)
        with open(f'{working_data_path}/Y_test.npy', 'rb') as f:
            y_test = np.load(f)
        with open(f'{working_data_path}/X_val.npy', 'rb') as f:
            X_val = np.load(f)
        with open(f'{working_data_path}/Y_val.npy', 'rb') as f:
            y_val = np.load(f)

    one_hot_labels = sorted(list(pathology_dict.values()))

    weights = compute_class_weight(class_weight = 'balanced', classes= one_hot_labels, y = y_val)
    print(one_hot_labels)
    print(weights)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output= False)
    enc.fit(y_train)

    y_test = enc.transform(y_test)
    y_train = enc.transform(y_train)
    y_val = enc.transform(y_val)

    return X_train, X_val, X_test, y_train, y_val, y_test, weights


class EEG_Dataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        with open(self.X[idx], 'rb') as f:
            eeg = np.load(f).astype(np.float32)
        return eeg, self.Y[idx]


def EEG_Dataloaders(split_ratio = 0.8, batch_size = 512):
    X_train, X_val, X_test, y_train, y_val, y_test, weights = read_and_oneHotencode()

    
    train_data_set = EEG_Dataset(X_train, y_train)

    if weighted_random_sampling:
        X_train_weights = weights[np.argmax(y_train, axis = 1)]
        weighted_sampler = WeightedRandomSampler(X_train_weights, len(train_data_set))
        train = DataLoader(train_data_set, batch_size=batch_size, sampler= weighted_sampler, num_workers=10)
    else:
        train = DataLoader(train_data_set, batch_size=batch_size, shuffle= True, num_workers=10)

    
    val_data_set = EEG_Dataset(X_val, y_val)
    val = DataLoader(val_data_set, batch_size=batch_size, shuffle= True, num_workers=10)
    
    test_data_set = EEG_Dataset(X_test, y_test)
    test = DataLoader(test_data_set, batch_size=batch_size, shuffle= True, num_workers=10)

    return train, val, test, weights


if __name__ == '__main__':
    train_data_loader, val_data_loader, test_data_loader, weights  = EEG_Dataloaders()
    print(train_data_loader)