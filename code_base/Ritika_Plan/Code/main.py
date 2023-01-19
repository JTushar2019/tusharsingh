from training_loops import *
from models import *
from data_loaders import *
from global_variables import *
from EEG_reading import preprocess_whole_data
import torch
import sys
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


############### logging part
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y %H:%M:%S")
sys.stdout = open('/home/tusharsingh/code_base/Ritika_Plan/Code/tracking2.txt', "a")
print("date and time =", dt_string)
print(f'picked channels = {dicided_channels_name}')
preprocess_whole_data()
print('\n\nNOW TRAINING')
###############



device = 'cuda'
train_loader, val_loader, test_loader = EEG_Dataloaders()
model = nn.DataParallel(Model())
# model = Model()

# path = "/home/tusharsingh/code_base/Ritika_Plan/Data/parameters-normalVsBruxism_TAcc82.pt"
# if os.path.exists(path):
#     model.load_state_dict(torch.load(path))

# print(model)
model = train_model(model, train_loader, val_loader, device, lr = 1e-3, max_epoc=1000,patience=100)
torch.cuda.empty_cache()
print("Testing model")
score = test_model(model, test_loader, device)
print('-'*100)
print("\n\n\n\n")
torch.save(model.state_dict(), f'{data_folder_path}/../parameters-normalVsBruxism_TAcc{score:0.0f}.pt')
del model

