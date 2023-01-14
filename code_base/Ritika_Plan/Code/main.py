from training_loops import *
from models import *
from data_loaders import *
from global_variables import *
import torch
import sys
from datetime import datetime


############# logging part
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
sys.stdout = open('/home/tusharsingh/code_base/Ritika_Plan/Code/tracking.txt', "a")
print("\n\n\n\ndate and time =", dt_string)
#############


device = 'cuda'
train_loader, val_loader, test_loader = EEG_Dataloaders()
model = nn.DataParallel(Model())

# path = "/home/tusharsingh/code_base/Ritika_Plan/Data/best-model-parameters.pt"
# model.load_state_dict(torch.load(path))

print(model)
model = train_model(model, train_loader, val_loader, device, max_epoc=1000)
score = test_model(model, test_loader, device)


torch.save(model.state_dict(), f'{data_folder_path}/../best-model-parameters-at{dt_string}.pt')


del model
torch.cuda.empty_cache()  