import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from EEG_reading import preprocess_whole_data
from global_variables import *
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


# working_data_path = "/home/tusharsingh/code_base/Ritika_Plan/Data"
working_data_path = f'{data_folder_path}/..'


def train_val_test_split(pathology_dict, split_ratio=0.8):
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

    one_hot_labels = sorted(list(pathology_dict.values()))
    print(one_hot_labels)

    # weights = compute_class_weight(class_weight = 'balanced', classes= one_hot_labels, y = Y)


    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, train_size=split_ratio, random_state=123, stratify=Y)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
        train_size=split_ratio, random_state=123, stratify = y_train)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output= False)
    enc.fit(y_train)

    one_hot_labels = list(enc.categories_[0])
    
    y_test = enc.transform(y_test)
    y_train = enc.transform(y_train)
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


def EEG_Dataloaders(pathology_dict, split_ratio = 0.8, batch_size = 512):
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(pathology_dict, split_ratio)
    
    train_data_set = EEG_Dataset(X_train, y_train)
    train = DataLoader(train_data_set, batch_size=batch_size, shuffle= True, num_workers=10)
    
    val_data_set = EEG_Dataset(X_val, y_val)
    val = DataLoader(val_data_set, batch_size=batch_size, shuffle= True, num_workers=10)
    
    test_data_set = EEG_Dataset(X_test, y_test)
    test = DataLoader(test_data_set, batch_size=batch_size, shuffle= True, num_workers=10)

    return train, val, test


# if __name__ == '__main__':
#     X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split()
#     print(y_train.shape, y_val.shape, y_test.shape)
#     print(X_train.shape, X_val.shape, X_test.shape)
#     print(X_train[0])
#     train_data_loader, val_data_loader, test_data_loader  = EEG_Dataloaders()
#     print(train_data_loader)