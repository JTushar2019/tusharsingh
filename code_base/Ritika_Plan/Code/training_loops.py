import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, log_loss
import copy



def train_model(model, train_loader, val_loader, device, lr=1e-3, max_epoc=100, patience=30):

    best_model_wts = copy.deepcopy(model.state_dict())
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, foreach = True)
    loss = nn.CrossEntropyLoss(reduction='sum')
    best_loss = 1
    temp_patience = patience
    for ep in range(max_epoc):

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

        if ep % 10 == 0 or ep == (max_epoc - 1):
            print(f'\tEpoch:{ep}\n\tT.B_Acc_score:{training_acc:.3f}, V.B_Acc_score:{validation_acc:.5f}')
            print(f'\t\t\tT.Cross_Entr_loss:{training_loss:.5f}, V.Cross_Entr_loss:{validation_loss:.5f}\n')


        if validation_loss > best_loss :
            patience -= 1
            if patience <= 0:
                print('Early stopping :(')
                print(f'\tEpoch:{ep}\n\tT.B_Acc_score:{training_acc:.5f}, V.B_Acc_score:{validation_acc:.5f}')
                print(f'\t\t\tT.Cross_Entr_loss:{training_loss:.5f}, V.Cross_Entr_loss:{validation_loss:.5f}\n')
                break
        else:
            best_loss = validation_loss
            patience = temp_patience
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


def test_model(model, test_loader, device):
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
        # correct += (pred.argmax(dim=1) == Y.argmax(dim=1)).sum().item()
        # metrics
        predictedY.extend(pred.argmax(dim=1).detach().cpu().tolist())
        realY.extend(Y.argmax(dim=1).detach().cpu().tolist())


    accuracy = balanced_accuracy_score(realY, predictedY)
    total_loss = total_loss / len(test_loader.dataset)
    # accuracy = correct / len(test_loader.dataset)
    print(f'Test_DATA: Cross_Entr_loss: {total_loss:.5f}, T_B_Acc_score: {accuracy:.5f}')

    return accuracy * 100
