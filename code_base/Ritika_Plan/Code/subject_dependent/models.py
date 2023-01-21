import torch.nn as nn
import torch
from global_variables import *

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        inputchannel = len(decided_channels)
        num_classes = len(pathology_dict)
        sfeq = sampling_frequency
        
        self.seq_layer1 = nn.Sequential(
            nn.Conv1d(inputchannel, 128, sfeq, sfeq // 16),
            nn.MaxPool1d(4, 4),

            nn.Conv1d(128, 64, 8, 1),
            nn.MaxPool1d(4, 4),
            
            nn.Conv1d(64, 32, 8, 1),
            nn.MaxPool1d(4, 4),

            nn.Flatten()

        )
        self.seq_layer2 = nn.Sequential(
            nn.Conv1d(inputchannel, 64, sfeq * 2, stride = sfeq // 4),
            nn.MaxPool1d(4, 4),

            nn.Conv1d(64, 32, 4),
            nn.MaxPool1d(4, 4),

            nn.Conv1d(32, 16, 4),
            nn.MaxPool1d(2, 2),

            nn.Flatten()
            
         )

        self.fullyConnected = nn.Sequential(
            nn.Dropout(0.3),
            # nn.LazyLinear(512),
            nn.Linear(176,64),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(64,num_classes),
            nn.Sigmoid(),
            nn.Softmax(dim=-1)            
        )
    
        # Initialization
        self.seq_layer1.apply(self.weights_init)
        self.seq_layer2.apply(self.weights_init)
        self.fullyConnected.apply(self.weights_init)


    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)


    def forward(self, x):
        x2 = x.detach().clone()
        x1 = self.seq_layer1(x)
        x2 = self.seq_layer2(x2)

        x = torch.cat((x1, x2),1)
        x = self.fullyConnected(x)
        return x



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import torchinfo
    model = Model()
    x = torchinfo.summary(model, (len(decided_channels), 7680), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "kernel_size"), verbose = 0)
    print(x)
    print(model)


    
