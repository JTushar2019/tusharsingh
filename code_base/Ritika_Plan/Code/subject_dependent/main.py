from training_loops import *
from models import *
from data_loaders import *
from global_variables import *
from EEG_reading import preprocess_whole_data
import torch
import sys
import os, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime

logging_file = open(logging_part, "a")
sys.stdout = logging_file

def evaluate(pathology_dict, decided_channels):

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y %H:%M:%S")
    print("date and time =", dt_string)
    print(f'picked channels = {decided_channels}')

    preprocess_whole_data(decided_channels, pathology_dict)

    print('\n\nTRAINING')

    with open(f'{saved_model_path}/sampling_frequency.pickle', 'rb') as handle:
        sfeq = int(pickle.load(handle))
        print(f'sampling_freq = {sfeq}')

    device = 'cuda'
    train_loader, val_loader, test_loader = EEG_Dataloaders(pathology_dict, split_ratio = 0.8, batch_size = 512)
    model = Model(sfeq, decided_channels, pathology_dict)

    # print(model)
    model = train_model(model, pathology_dict, decided_channels, train_loader, val_loader, device, lr = 1e-3, max_epoc=300, patience = 60)

    print("Testing model")
    score, loss = test_model(model,pathology_dict, decided_channels, test_loader, device)
    print('-'*100)
    print("\n\n\n\n")

    one_hot_labels = sorted(list(pathology_dict.values()))
    name = "_vs_".join(one_hot_labels) + '_' + "_".join(decided_channels)
    torch.save(model.state_dict(), f'{saved_model_path}/SD_params_{name}_{score:0.0f}.pt')

    temp_model = f'{saved_model_path}/Temp_params_{name}.pt'
    os.remove(temp_model)

    torch.cuda.empty_cache()
    del model


pathology_dict = {
    'brux': 'BRUX',
    'ins': 'INS', 
    # 'n': 'Normal',         
    'narco':  'NARCO', 
    'nfle': 'NFLE',
    'plm': 'PLM',  
    'rbd': 'RBD',  
    'sdb': 'SDB'
}


for key, value in pathology_dict.items():
    temp = dict()
    temp['n'] = 'Normal'
    temp[key] = value

    for each in channels[key]:
        decided_channels = [each]
        evaluate(temp, decided_channels)


logging_file.close()