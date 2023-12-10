
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class transformer_construction(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_heads):
        super(transformer_construction, self).__init__()

        self.num_heads = num_heads
        self.Query_lists = nn.ModuleList()
        self.Key_lists = nn.ModuleList()

        for i in range(num_heads):
            self.Query_lists.append(nn.Linear(input_dimension, hidden_dimension))
            self.Key_lists.append(nn.Linear(input_dimension, hidden_dimension))


    def forward(self, X):
        # print('construction input is', X.size())
        relation_lists = []

        for i in range(self.num_heads):
            Query_i = self.Query_lists[i](X)
            Query_i = F.leaky_relu(Query_i)

            Key_i = self.Key_lists[i](X)
            Key_i = F.leaky_relu(Key_i)

            T_rela_i = torch.bmm(Query_i, torch.transpose(Key_i,-1, -2)) ## size is lists of (bs, N, N)

            relation_lists.append(T_rela_i)

        relation_lists = torch.stack(relation_lists, 1)

        relation_lists = torch.mean(relation_lists, 1)
        return relation_lists


def Dot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)

    Adj = torch.bmm(node_features, node_features_1)

    eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()

    eyes_like_inf = eyes_like*1e8

    Adj = F.leaky_relu(Adj-eyes_like_inf)

    Adj = F.softmax(Adj, dim = -1)

    Adj = Adj+eyes_like

    return Adj
