import torch
import torch.nn as nn



class LCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LCNN, self).__init__()

        self.l1 = nn.Conv1d(input_dim, hidden_dim, 3)
        self.r1 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.l2 = nn.Conv1d(hidden_dim * 3, hidden_dim, 3)
        self.r2 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        # self.l3 = nn.Conv1d(hidden_dim * 3, hidden_dim, 3)
        # self.r3 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.l3 = nn.Conv1d(hidden_dim * 3, hidden_dim, 3)

        self.pool = nn.MaxPool1d(3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pool(self.l1(x.permute(0, 2, 1)))
        x = torch.cat((self.r1(x.permute(0, 2, 1))[0].permute(0, 2, 1), x), dim=1)
        x = self.pool(self.l2(x))
        x = torch.cat((self.r2(x.permute(0, 2, 1))[0].permute(0, 2, 1), x), dim=1)
        # x = self.pool(self.l3(x))
        # x = torch.cat((self.r3(x.permute(0, 2, 1))[0].permute(0, 2, 1), x), dim=1)
        x = self.pool(self.l3(x))

        x = self.global_pool(x).squeeze(2)

        return x