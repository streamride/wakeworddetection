import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.rnn = nn.LSTM(128, 256, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64,1)
        )
        # self.fc1 = nn.Linear(256, 128)
        # self.fc = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm = nn.LayerNorm(128)

    # def init_hidden(self, batch_size):
        # return 

    def forward(self, x):
        x = x.squeeze(1)
        x = self.layer_norm(x)
        # print(x.shape)
        # x = x.transpose(0,1)
        # x = x.transpose(1,2)
        
        # print(x.shape)
        x, (hidden_last, cell_last) = self.rnn(x)
        hidden_last = self.dropout(hidden_last)
        # print(x.shape)
        # print(hidden_last.shape)
        hidden_last = hidden_last.squeeze(0)
        x = self.classifier(hidden_last)
        # print(x.shape)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(128)

        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc1 = nn.Linear(32*14*14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    
    def forward(self, x):
        # print(x.shape)
        batch_size, channels, _, _ = x.shape
        x = self.layer_norm(x)
        # print(x.shape)
        x = self.cnn_block1(x)
        
        # print(x.shape)
        x = self.cnn_block2(x)
        # print(x.shape)
        x = self.cnn_block3(x)
        # print(x.shape)
        
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # print(x.shape)
        x = self.output(x)
        return x