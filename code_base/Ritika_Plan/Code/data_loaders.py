import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from EEG_reading import preprocess_whole_data
from feature_extraction import *
from global_variables import *
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder


# working_data_path = "/home/tusharsingh/code_base/Ritika_Plan/Data"
working_data_path = f'{data_folder_path}/..'


def train_val_test_split(split_ratio=0.8):
    if not os.path.exists(f'{working_data_path}/X.npy'):
        X, Y = preprocess_whole_data()
    else:
        with open(f'{working_data_path}/X.npy', 'rb') as f:
            X = np.load(f)
        with open(f'{working_data_path}/Y.npy', 'rb') as f:
            Y = np.load(f)

    Y = Y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, train_size=split_ratio, random_state=123, stratify=Y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, train_size=split_ratio, random_state=123, stratify=y_train)

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output= False)
    enc.fit(Y)

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)
    y_val = enc.transform(y_val)

    return X_train, X_val, X_test, y_train, y_val, y_test


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


def EEG_Dataloaders(split_ratio = 0.8, batch_size = 128):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(split_ratio)

    train_data_loader = EEG_Dataset(X_train, y_train)
    train = DataLoader(train_data_loader, batch_size=batch_size,
                       shuffle=True, num_workers=16)
    
    val_data_loader = EEG_Dataset(X_val, y_val)
    val = DataLoader(val_data_loader, batch_size=batch_size,
                     shuffle=True, num_workers=16)
    
    test_data_loader = EEG_Dataset(X_test, y_test)
    test = DataLoader(test_data_loader, batch_size=batch_size,
                      shuffle=True, num_workers=16)

    return train, val, test


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split()
    print(y_train.shape, y_val.shape, y_test.shape)
    print(X_train.shape, X_val.shape, X_test.shape)
    print(X_train[0])
    train_data_loader, val_data_loader, test_data_loader = EEG_Dataloaders()
    print(train_data_loader)
    print(dir(val_data_loader.dataset['Y']))
