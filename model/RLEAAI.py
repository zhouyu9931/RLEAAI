import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from model.rcca import RCCAModule
from model.lcnn import LCNN



class rleModel(nn.Module):
    def __init__(self, emb_dim=1280, input_dim=128, hidden_dim=32):
        super(rleModel, self).__init__()

        self.ab_fc = nn.Sequential(
            nn.Linear(emb_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
        )
        self.ag_fc = nn.Sequential(
            nn.Linear(emb_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
        )

        self.lccn = LCNN(input_dim=input_dim, hidden_dim=hidden_dim)

        self.ab_cca = RCCAModule(in_channels=emb_dim * 4)
        self.ag_cca = RCCAModule(in_channels=emb_dim * 4)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.3),
            nn.Linear(16, 1)
        )


    def forward(self, ab_emb, ag_emb, ab_cks, ag_cks):
        ab = self.ab_fc(ab_emb)
        s1 = self.lccn(ab)
        ab_cc = self.ab_cca(ab_cks)
        ab = s1 + ab_cc + (s1 * ab_cc) * self.alpha

        ag = self.ag_fc(ag_emb)
        s2 = self.lccn(ag)
        ag_cc = self.ag_cca(ag_cks)
        ag = s2 + ag_cc + (s2 * ag_cc) * self.beta
        s = torch.mul(ab, ag)

        x = self.fc(s)
        return x


# net = myModel(1280,128, 32)
# print(sum(p.numel() for p in net.parameters()))
# x1 = torch.randn(3, 200, 1280)
# x2 = torch.randn(3, 780, 1280)
# x3 = torch.randn(3, 5120, 20, 20)
# x4 = torch.randn(3, 5120, 20, 20)
# # flops, params = profile(net, inputs=(x1,x2,x3,x4))
# # flops, params = clever_format([flops, params], "%.3f")
# # print(f"FLOPs: {flops}")
# # print(f"Params: {params}")
# start_time = time.time()
# y = net(x1, x2, x3, x4)
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
# print(y.shape)
# print(y)


# y= F.sigmoid(y)
# print(y)
# # y = y[:,1]
# print(y.shape, y[:,1])
# label = torch.tensor([0, 1, 0])
# onehot_label = F.one_hot(label, num_classes=2)
# # label = label.unsqueeze(1)
# print(y, onehot_label)
# print(y.shape, onehot_label.shape)
# crition = torch.nn.BCELoss()
# loss = crition(y, onehot_label.float())
# print(loss)