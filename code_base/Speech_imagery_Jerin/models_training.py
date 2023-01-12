import copy
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def model_maker(num_classes):
    Jerrin_model = models.resnet18(weights='DEFAULT')
    for param in Jerrin_model.parameters():
        param.requires_grad = False
    Jerrin_model.fc = nn.Sequential(
        nn.Linear(Jerrin_model.fc.in_features, 128),
        nn.RReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(128, 64),
        nn.RReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(64, num_classes),
        nn.Softmax(dim=1)
    )
    # return nn.DataParallel(Jerrin_model)
    return Jerrin_model


def train_model(model, train_loader, val_loader, device, max_epoc=100, patience=30):

    best_model_wts = copy.deepcopy(model.state_dict())
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss = nn.CrossEntropyLoss(reduction = 'sum')
    best_loss = 1
    temp_patience = patience
    for ep in range(max_epoc):

        training_loss = 0
        correct = 0
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss_batch = loss(pred, Y)
            correct += (pred.argmax(dim=1) == Y).sum().item()
            training_loss += loss_batch.item()
            loss_batch.backward()
            optimizer.step()

        training_loss = training_loss / len(train_loader.dataset)
        training_acc = correct / len(train_loader.dataset)

        val_loss = 0
        correct = 0
        model.eval()
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss_batch = loss(pred, Y)
            correct += (pred.argmax(dim=1) == Y).sum().item()
            val_loss += loss_batch.item()

        validation_acc = correct / len(val_loader.dataset)
        validation_loss = val_loss / len(val_loader.dataset)

        if ep % 5 == 0 or ep == (max_epoc - 1):
            print(f'\t epoch:{ep}, T.acc:{training_acc*100:.3f}, V.acc:{validation_acc*100:.3f}')
            print(f'\t\t T.loss:{training_loss:.5f}, V.loss:{validation_loss:.5f}')


        if validation_loss > best_loss :
            patience -= 1
            if patience <= 0:
                print('Early stopping :(')
                print(f'\t epoch:{ep}, T.acc:{training_acc*100:.3f}, V.acc:{validation_acc*100:.3f}')
                print(f'\t\t T.loss:{training_loss:.5f}, V.loss:{validation_loss:.5f}')
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
    correct = 0
    loss = nn.CrossEntropyLoss()
    total_loss = 0
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        total_loss += loss(pred, Y) * X.size(0)
        correct += (pred.argmax(dim=1) == Y).sum().item()

    total_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test_fold: Tloss: {total_loss:.5f}, Tacc: {accuracy*100:.3f}')
    return accuracy
