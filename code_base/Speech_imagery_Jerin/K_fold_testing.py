import argparse
from models_training import *
from data_loaders import *
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import torch


def K_fold_evaluation(type, subj_no, device, kfolds=10, batch_size = 64, random_seed=123):
    X, Y = load_EEG(type, subj_no)

    skf = StratifiedKFold(
        n_splits=10, random_state=random_seed, shuffle=True)

    score = [0]*kfolds
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        print(f"------------------------fold {i}------------------------")
        train_X, val_X, train_Y, val_Y = train_test_split(
            X[train_index], Y[train_index], test_size=0.15, stratify= Y[train_index], random_state=random_seed)
        test_X, test_Y = X[test_index], Y[test_index]

        train_loader = EEG_Dataloader(train_X, train_Y, type, batch_size, test = False)
        val_loader = EEG_Dataloader(val_X, val_Y, type, batch_size, test = False)
        test_loader = EEG_Dataloader(test_X, test_Y, type)

        model = model_maker(len(numeric_labels[type]))
        model = train_model(model, train_loader, val_loader, device, max_epoc=150)
        score[i] = test_model(model, test_loader, device)
        del model
        torch.cuda.empty_cache()  
    return np.mean(score), np.std(score)


if __name__ == "__main__":
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    file_path = 'results.txt'
    import sys
    sys.stdout = open(file_path, "a")
    print("\n\n\n\ndate and time =", dt_string)
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='Long_words')
    parser.add_argument('--subj_no', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--kfolds', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=123)
    args = parser.parse_args()
    print(args)

    subjects = [2,3,6,7,9,11]
    for each in subjects[1:]:
        print()
        print("="*150)
        mean, variance = K_fold_evaluation(args.type, each, args.device, \
            args.kfolds, args.batch_size, args.random_seed)
        print(f"{each} K-fold mean: ", mean*100)
        print(f"{each} K-fold variance: ", variance*100)
    






