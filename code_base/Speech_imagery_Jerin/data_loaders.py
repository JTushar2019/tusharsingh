from pre_processing import *
from global_variables import numeric_labels
from feature_extr import EEG_Image
from torch.utils.data import Dataset, DataLoader


class EEG_Dataset(Dataset):

    def __init__(self, X, Y, type, test = False):
        if test:
            X, Y = get_test_preprocessed_data(
                X,Y, test_pre_process, EEG_Image)
        else:
            X, Y = get_train_preprocessed_data(
                X,Y, train_pre_process, EEG_Image)
        self.X = X
        self.Y = [numeric_labels[type][y] for y in Y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        with open(self.X[idx], 'rb') as f:
            image = np.load(f).transpose(2, 0, 1).astype(np.float32)
        return image, self.Y[idx]


def EEG_Dataloader(X, Y,  type, batch_size = 4, test = True):
    return DataLoader(
        EEG_Dataset(X, Y, type),
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
    )