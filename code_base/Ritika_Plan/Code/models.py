import torch.nn as nn
from global_variables import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        inputchannel = len(decided_channels)
        num_classes = len(pathology_dict)
        sfeq = sampling_frequency
        self.dim = 0
        
        self.seq_layer1 = nn.Sequential(
            nn.Conv1d(inputchannel, 64, sfeq // 2, sfeq // 16),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(8, 8),
            nn.Dropout(p = 0.5),

            nn.Conv1d(64, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(38,50),
            nn.ReLU(),
            nn.Linear(50,num_classes),
            nn.Softmax(dim=-1)

        )

    def forward(self, x):
        x = self.seq_layer1(x)
        return x



if __name__ == '__main__':
    import torchinfo
    model = Net()
    x = torchinfo.summary(model, (3, 7680), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "kernel_size"), verbose = 0)
    print(x)
    print(model)


    
