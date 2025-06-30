import torch.nn as nn


class lstm(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.layer_1 = nn.Linear(hidden_size,hidden_size//2)
        self.layer_2 = nn.Linear(hidden_size//2,hidden_size//4)
        self.out = nn.Linear(hidden_size//4,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x