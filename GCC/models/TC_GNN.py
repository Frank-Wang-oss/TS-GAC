import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer



class TC(nn.Module):
    def __init__(self, configs, args, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        # self.timestep = configs.timesteps

        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, self.num_channels // 2),
            nn.BatchNorm1d(self.num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_channels // 2, self.num_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        ### features in graph have size (bs, time_length, num_nodes, feature_dimension)
        # print(features_aug1.size())
        batch, seq_len, num_nodes, feature_dimension = features_aug1.size()

        ## choice 1:
        features_aug1 = torch.transpose(features_aug1, 1,2)
        z_aug1 = torch.reshape(features_aug1, [batch * num_nodes, seq_len, feature_dimension])

        features_aug2 = torch.transpose(features_aug2, 1, 2)
        z_aug2 = torch.reshape(features_aug2, [batch * num_nodes, seq_len, feature_dimension])


        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch*num_nodes, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch*num_nodes, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]
        # print('forward_seq', forward_seq.size())

        c_t = self.seq_transformer(forward_seq)
        # print('c_t', c_t.size())

        pred = torch.empty((self.timestep, batch*num_nodes, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        nce /= -1. * batch * num_nodes * self.timestep
        # nce /= -1. * batch * self.timestep
        out = self.projection_head(c_t)
        # print(out.size())
        out = torch.reshape(out, [batch, num_nodes, -1])


        # return nce, torch.reshape(out, [batch, -1])
        return nce, out
