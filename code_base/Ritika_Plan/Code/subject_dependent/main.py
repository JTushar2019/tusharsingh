from training_loops import *
from models import *
from data_loaders import *
from global_variables import *
from EEG_reading import preprocess_whole_data
import torch
import sys
import os
from datetime import datetime
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


logging_file = open('/home/tusharsingh/code_base/Ritika_Plan/Data/Progress_tracking.txt', "a")
sys.stdout = logging_file

print('\n\nTRAINING')

one_hot_labels = sorted(list(pathology_dict.values()))

device = 'cuda'
train_loader, val_loader, test_loader = EEG_Dataloaders()
# model = nn.DataParallel(Model())
model = Model()

# print(model)
model = train_model(model, train_loader, val_loader, device, lr = 1e-3, max_epoc=300, patience=40)

print("Testing model")
score, loss = test_model(model, test_loader, device)
print('-'*100)
print("\n\n\n\n")
torch.save(model.state_dict(), f'{saved_model_path}/SD_params_{"_vs_".join(one_hot_labels)}_{score:0.0f}.pt')

torch.cuda.empty_cache()
del model

logging_file.close()