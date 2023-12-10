
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import copy

class Feature_Extractor_LSTM(nn.Module):
    def __init__(self,  input_dim, num_hidden, output_dim):
        super(Feature_Extractor_LSTM, self).__init__()
        self.input_dim = input_dim

        self.bi_lstm1 = nn.LSTM(input_size=self.input_dim,
                                hidden_size=num_hidden,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)
        self.drop1 = nn.Dropout(p=0.2)

        self.bi_lstm2 = nn.LSTM(input_size=num_hidden,
                                hidden_size=num_hidden*2,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)
        self.drop2 = nn.Dropout(p=0.2)

        self.bi_lstm3 = nn.LSTM(input_size=num_hidden*2,
                                hidden_size=output_dim,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

        self.drop3 = nn.Dropout(p=0.2)


    # Defining the forward pass
    def forward(self, x):
        ### size of x is (time_length, Bs, dimension)
        x, hidden = self.bi_lstm1(x)
        x_split = tr.split(x, (x.shape[2] // 2), 2)
        x = x_split[0] + x_split[1]
        x, hidden = self.bi_lstm2(x)
        x_split = tr.split(x, (x.shape[2] // 2), 2)
        x = x_split[0] + x_split[1]
        x = self.drop2(x)

        x2, hidden = self.bi_lstm3(x)
        x2_presp = x2

        x2_split = tr.split(x2_presp, (x2_presp.shape[2] // 2), 2)
        x2 = x2_split[0] + x2_split[1]
        x2 = self.drop3(x2)

        return F.leaky_relu(x2)


class Feature_extractor_1DCNN(nn.Module):
    def __init__(self, input_channels, num_hidden, embedding_dimension, kernel_size = 8, stride = 1, dropout = 0.35):
        super(Feature_extractor_1DCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )


    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        return x


class Feature_extractor_1DCNN_tiny(nn.Module):
    def __init__(self, input_channels, num_hidden, embedding_dimension, kernel_size = 3, stride = 1, dropout = 0.35):
        super(Feature_extractor_1DCNN_tiny, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # print()

    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        # print(x.size())
        return x
