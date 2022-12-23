from pre_processing import load_EEG, get_preprocessed_data, train_pre_process, test_pre_process, numeric_labels
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
import warnings
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models
torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)
cudnn.benchmark = True


def K_fold_evaluation(type, subj_no, kfolds=10, random_seed=123):
    Y = np.array([x.split("_")[0] for x in X])
    X = np.array(load_EEG(type, subj_no))

    skf = StratifiedKFold(
        n_splits=10, random_state=random_seed, shuffle=True)

    score = [0]*kfolds
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        train_X, train_Y = get_preprocessed_data(
            X[train_index], train_pre_process)
        test_X, test_Y = get_preprocessed_data(
            X[test_index], test_pre_process)

        train_X, val_X, train_Y, val_Y = train_test_split(
            train_X, train_Y, test_size=0.15, stratify= True, random_state=random_seed)

        train_dataset = EEG_Dataset(train_X, train_Y, type)
        val_dataset = EEG_Dataset(val_X, val_Y, type)
        test_dataset = EEG_Dataset(test_X, test_Y, type)

        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=True, num_workers=4)
        test_loader = DataLoader(
            test_dataset, batch_size=10, shuffle=True, num_workers=4)

        model = train_model(train_loader,val_loader)
        score[i] = test_model(test_loader, model)    
    return np.mean(score), np.std(score)


class EEG_Dataset(Dataset):

    def __init__(self, X, Y, type):
        self.X = X
        self.Y = [numeric_labels(y) for y in Y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        with open(self.X[idx], 'rb') as f:
            image = np.load(f).transpose(2, 0, 1).astype(np.float32)
        return image, self.Y[idx]




# dataset = 'Vowels'
# subject = 12

# output_layer_size = len(words[dataset])

# dataset = EEG_Dataset(
#     eeg_image_folder_path[dataset], subject, words[dataset])

# train_dataset, val_dataset = random_split(
#     dataset, [0.85, 0.15], generator=torch.Generator().manual_seed(42))

# EEG_image_datasets = {'train': train_dataset, 'valid': val_dataset}
# dataloaders = {x: DataLoader(
#     EEG_image_datasets[x], batch_size=10, shuffle=True, num_workers=12) for x in ['train', 'valid']}
# dataset_sizes = {x: len(EEG_image_datasets[x]) for x in ['train', 'valid']}

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# class NeuralNetwork(nn.Module):
#     def __init__(self, number_of_words):
#         super(NeuralNetwork, self).__init__()
#         self.ResNet = models.resnet50(pretrained=True)
#         for param in self.ResNet.parameters():
#             param.requires_grad = False
#         self.fc1 = nn.Linear(self.ResNet.fc.in_features, 128)
#         self.ResNet.fc = nn.Identity()
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, number_of_words)
#         self.number_of_words = number_of_words

#     def forward(self, x):
#         x = self.ResNet(x)
#         x = self.fc1(x)
#         x = nn.functional.leaky_relu(x)
#         x = nn.functional.dropout(x, 0.3)
#         x = self.fc2(x)
#         x = nn.functional.leaky_relu(x)
#         x = nn.functional.dropout(x, 0.3)
#         x = self.fc3(x)
#         x = nn.functional.softmax(x, dim = 1)
#         return x


# model = NeuralNetwork(output_layer_size).to(device=device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-5)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f'\nEpoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'valid']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 # print( loss.item(), outputs, labels,running_loss)
#                 running_corrects += torch.sum(preds == labels.data)
#             if phase == 'train':
#                 scheduler.step()
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             # deep copy the model
#             if phase == 'valid' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#     time_elapsed = time.time() - since
#     print(
#         f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:4f}')

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model


# model_ft = train_model(model, criterion, optimizer,
#                        exp_lr_scheduler, num_epochs=300)
