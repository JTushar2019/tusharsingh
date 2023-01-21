from training_loops import *
from models import *
from data_loaders import *
from global_variables import *
from EEG_reading import preprocess_whole_data
import torch
import sys
from datetime import datetime
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


############### logging part
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y %H:%M:%S")
sys.stdout = open('/home/tusharsingh/code_base/Ritika_Plan/Code/tracking3.txt', "a")
print("date and time =", dt_string)
print(f'picked channels = {dicided_channels_name}')



preprocess_whole_data()
print('\n\nNOW TRAINING')
###############

one_hot_labels = sorted(list(pathology_dict.values()))


device = 'cuda'
train_loader, val_loader, test_loader, class_weight = EEG_Dataloaders()
# model = nn.DataParallel(Model())
model = Model()

# print(model)
model = train_model(model, train_loader, val_loader, class_weight, device, lr = 1e-3, max_epoc=500,patience=40)
torch.cuda.empty_cache()
print("Testing model")
score = test_model(model, test_loader, device)
print('-'*100)
print("\n\n\n\n")
torch.save(model.state_dict(), f'{saved_model_path}/SI_params_{"_vs_".join(one_hot_labels)}_{score:0.0f}.pt')
del model

