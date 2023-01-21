from training_loops import *
from models import *
from data_loaders import *
from global_variables import *
from EEG_reading import preprocess_and_record
import torch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def final_running(max_epoc=1000):
    logging_file = open(tracking_file_path, "a")
    temp = sys.stdout
    sys.stdout = logging_file

    one_hot_labels = sorted(list(pathology_dict.values()))

    device = 'cuda'
    train_loader, val_loader, test_loader, weights = EEG_Dataloaders()
    # model = nn.DataParallel(Model())
    model = Model()

    print('\n\nTRAINING')
    model = train_model(model, train_loader, val_loader, weights, device, lr = 1e-3, max_epoc= max_epoc, patience=50)

    print("Testing model")
    score, loss = test_model(model, test_loader, weights, device)
    torch.save(model.state_dict(), f'{saved_model_path}/SD_params_{"_vs_".join(one_hot_labels)}_{score:0.0f}.pt')
    torch.cuda.empty_cache()
    del model

    print('-'*100)
    print("\n\n\n\n")
    sys.stdout = temp
    logging_file.close()


if __name__ == "__main__":
    final_running(max_epoc=200)
