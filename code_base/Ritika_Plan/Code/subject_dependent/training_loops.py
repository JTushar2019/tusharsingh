import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import copy,os
from global_variables import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch


def train_model(model,pathology_dict, decided_channels, train_loader, val_loader, device, lr=1e-3, max_epoc=100, patience=30):

    one_hot_labels = sorted(list(pathology_dict.values()))
    pic_name = "_vs_".join(one_hot_labels) + '_' + "_".join(decided_channels)
    
    temp_model = f'{saved_model_path}/Temp_params_{pic_name}.pt'
    
    best_loss = 10
    
    # warm start
    if os.path.exists(temp_model):
        model.load_state_dict(torch.load(temp_model))
        best_loss = test_model(model, pathology_dict, decided_channels, val_loader, device)[1]


    train_loss_track = []
    train_acc_track = []
    val_loss_track = []
    val_acc_track = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, foreach = True, amsgrad= True)
    loss = nn.CrossEntropyLoss(reduction='sum')
    temp_patience = patience

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10, verbose = True)
    for ep in range(1, max_epoc + 1):

        training_loss = 0
        # correct = 0
        model.train()

        realY = []
        predictedY = []
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss_batch = loss(pred, Y)
            # correct += (pred.argmax(dim=1) == Y.argmax(dim=1)).sum().item()

            predictedY.extend(pred.argmax(dim=1).detach().cpu().tolist())
            realY.extend(Y.argmax(dim=1).detach().cpu().tolist())

            training_loss += loss_batch.item()
            loss_batch.backward()
            optimizer.step()

        training_acc = balanced_accuracy_score(realY, predictedY)
        training_loss = training_loss / len(train_loader.dataset)
        train_loss_track.append(training_loss)
        train_acc_track.append(training_acc)

        val_loss = 0
        # correct = 0
        model.eval()

        realY = []
        predictedY = []
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            loss_batch = loss(pred, Y)
            # correct += (pred.argmax(dim=1) == Y.argmax(dim=1)).sum().item()
            val_loss += loss_batch.item()
            
            predictedY.extend(pred.argmax(dim=1).detach().cpu().tolist())
            realY.extend(Y.argmax(dim=1).detach().cpu().tolist())


        validation_acc = balanced_accuracy_score(realY, predictedY)
        validation_loss = val_loss / len(val_loader.dataset)
        
        scheduler.step(validation_loss)
        
        val_loss_track.append(validation_loss)
        val_acc_track.append(validation_acc)

        if ep % 50 == 0 or ep == max_epoc:
            print(f'\tEpoch:{ep}\n\t\tT.B_Acc_score:{training_acc:.5f},     V.B_Acc_score:{validation_acc:.5f}')
            print(f'\t\tT.Cross_Entr_loss:{training_loss:.5f}, V.Cross_Entr_loss:{validation_loss:.5f}\n')


        if validation_loss > best_loss :
            patience -= 1
            if patience <= 0:
                print('Early stopping :(')
                model.load_state_dict(best_model_wts)
                print(f'\tEpoch:{ep}\n\t\tT.B_Acc_score:{training_acc:.5f}, V.B_Acc_score:{validation_acc:.5f}')
                print(f'\t\tT.Cross_Entr_loss:{training_loss:.5f}, V.Cross_Entr_loss:{validation_loss:.5f}\n')
                break
        else:
            best_loss = validation_loss
            patience = temp_patience
            best_model_wts = copy.deepcopy(model.state_dict())                
            # torch.save(best_model_wts, f'{saved_model_path}/Temp_params_{pic_name}.pt')


    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Training Graph')

    ax1.plot(train_loss_track, '-g', label='Training_loss')
    ax1.plot(val_loss_track, '-r', label='Validation_loss')
    ax2.plot(train_acc_track, '-g', label='Train_Acc')
    ax2.plot(val_acc_track, '-r', label='Validation_Acc')

    ax1.legend(fancybox=True)
    ax2.legend(fancybox=True)

    plt.tight_layout()

    print(f'Training_Stats saved as {pic_name}\n')

    plt.savefig(f'{confusion_matrix_path}/stats_{pic_name}.png')
    plt.close()


    return model


def test_model(model,pathology_dict, decided_channels, test_loader, device):

    model.to(device)
    model.eval()
    loss = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0
    # correct = 0

    realY = []
    predictedY = []
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        total_loss += loss(pred, Y)
        predictedY.extend(pred.argmax(dim=1).detach().cpu().tolist())
        realY.extend(Y.argmax(dim=1).detach().cpu().tolist())


    accuracy = balanced_accuracy_score(realY, predictedY)
    total_loss = total_loss / len(test_loader.dataset)
    # accuracy = correct / len(test_loader.dataset)
    print(f'Test_DATA: Cross_Entr_loss: {total_loss:.5f}, T_B_Acc_score: {accuracy:.5f}')


    cm = confusion_matrix(realY, predictedY)
    
    one_hot_labels = sorted(list(pathology_dict.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=one_hot_labels)
    disp.plot()

    pic_name = "_vs_".join(one_hot_labels) + "_".join(decided_channels)
    print(f'confusion_matrix saved as {pic_name}')
    plt.savefig(f'{confusion_matrix_path}/{pic_name}.png', bbox_inches = 'tight')
    # plt.show()

    return accuracy * 100, total_loss
