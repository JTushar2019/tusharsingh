from training_loops import *
from models import *
from data_loaders import *
from global_variables import *
from EEG_reading import preprocess_whole_data
import torch
import sys
from datetime import datetime


# ############# logging part
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y %H:%M:%S")
sys.stdout = open('/home/tusharsingh/code_base/Ritika_Plan/Code/tracking.txt', "a")
print("\n\n\n\ndate and time =", dt_string)
print(f'picked channels = {dicided_channels_name}')
preprocess_whole_data()
print('\n\nNOW TRAINING')
#############


device = 'cuda:01'
train_loader, val_loader, test_loader = EEG_Dataloaders()
# model = nn.DataParallel(Model())
model = Model()

# path = "/home/tusharsingh/code_base/Ritika_Plan/Data/best-model-parameters.pt"
# model.load_state_dict(torch.load(path))

print(model)
model = train_model(model, train_loader, val_loader, device, max_epoc=100,patience=50)
torch.cuda.empty_cache()
score = test_model(model, test_loader, device)


torch.save(model.state_dict(), f'{data_folder_path}/../best-model-parameters-at{dt_string}.pt')


del model