from .Graph_construction import *
from .Feature_extractor import *
from functools import wraps
from .Augmentation import *
from collections import OrderedDict

class GCNLayer(nn.Module):

    def __init__(self, in_ft, out_ft, act='prelu', bias=True):

        super(GCNLayer, self).__init__()

        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else nn.ReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )


class Base_model(nn.Module):
    def __init__(self, configs, args):
        super(Base_model, self).__init__()


        indim_fea = configs.window_size
        hidden_fea = configs.hidden_channels
        outdim_fea = configs.final_out_channels
        num_node = configs.num_nodes
        num_classes = configs.num_classes
        self.nonlin_map = nn.Linear(indim_fea, indim_fea)

        self.hidden_dim = hidden_fea
        self.output_dim = outdim_fea
        self.time_length = configs.convo_time_length
        self.Time_Preprocessing = Feature_extractor_1DCNN_tiny(indim_fea,32,hidden_fea, configs.kernel_size, configs.stride, configs.dropout)

        self.MPNN = GCNLayer(hidden_fea, outdim_fea)

        self.logits = nn.Linear(self.time_length*outdim_fea*num_node, num_classes)

    def forward(self, X, self_supervised = False, num_remain = None):
        ## Size of X is (bs, time_length, num_nodes, dimension)
        bs, tlen, num_node, dimension = X.size()

        X = torch.transpose(X, 2, 1) # (bs, num_nodes, time_length, dimension)
        X = torch.reshape(X, [bs*num_node, tlen, dimension])
        X = self.nonlin_map(X)

        TD_input = tr.transpose(X, 1, 2)
        TD_output = self.Time_Preprocessing(TD_input)  ### size is (bs, out_dimension*num_node, tlen)
        TD_output = tr.transpose(TD_output, 1, 2)  ### size is (bs, tlen, out_dimension*num_node

        GC_input = tr.reshape(TD_output, [bs, -1, num_node, self.hidden_dim])

        GC_input = tr.reshape(GC_input, [-1, num_node, self.hidden_dim])
        Adj_input = Dot_Graph_Construction(GC_input)
        if self_supervised:
            Adj_input = changes_correlations(Adj_input, num_remain)

        GC_output = self.MPNN(GC_input, Adj_input)
        GC_output = tr.reshape(GC_output, [bs, -1, num_node, self.output_dim])

        logits_input = tr.reshape(GC_output, [bs, -1])

        logits = self.logits(logits_input)

        return logits, GC_output



class Base_model_woMW(nn.Module):
    def __init__(self, configs, args):
        super(Base_model_woMW, self).__init__()


        indim_fea = configs.window_size
        hidden_fea = configs.hidden_channels
        outdim_fea = configs.final_out_channels
        num_node = configs.num_nodes
        num_classes = configs.num_classes
        self.nonlin_map = nn.Linear(indim_fea, indim_fea)

        self.hidden_dim = hidden_fea
        self.output_dim = outdim_fea
        self.time_length = configs.convo_time_length
        self.Time_Preprocessing = Feature_extractor_1DCNN(indim_fea,32,hidden_fea, configs.kernel_size, configs.stride, configs.dropout)

        self.MPNN = GCNLayer(hidden_fea, outdim_fea)

        self.logits = nn.Linear(self.time_length*outdim_fea*num_node, num_classes)

    def forward(self, X, self_supervised = False, num_remain = None):
        ## Size of X is (bs, time_length, num_nodes, 1)
        bs, tlen, num_node, dimension = X.size()

        X = torch.transpose(X, 2, 1) # (bs, num_nodes, time_length, dimension)
        X = torch.reshape(X, [bs*num_node, tlen, dimension])
        X = self.nonlin_map(X)

        TD_input = tr.transpose(X, 1, 2)
        TD_output = self.Time_Preprocessing(TD_input)  ### size is (bs, out_dimension*num_node, tlen)
        TD_output = tr.transpose(TD_output, 1, 2)  ### size is (bs, tlen, out_dimension*num_node

        GC_input = tr.reshape(TD_output, [bs, -1, num_node, self.hidden_dim])

        GC_input = tr.reshape(GC_input, [-1, num_node, self.hidden_dim])
        Adj_input = Dot_Graph_Construction(GC_input)
        if self_supervised:
            Adj_input = changes_correlations(Adj_input, num_remain)

        GC_output = self.MPNN(GC_input, Adj_input)
        GC_output = tr.reshape(GC_output, [bs, -1, num_node, self.output_dim])

        logits_input = tr.reshape(GC_output, [bs, -1])

        logits = self.logits(logits_input)

        return logits, GC_output

class GCN_layer(nn.Module):
    def __init__(self, input_dimension, out_dimension):
        super(GCN_layer, self).__init__()

        self.nonlinear_FC = nn.Sequential(
            nn.Linear(input_dimension, out_dimension),
            nn.ReLU()
        )

    def forward(self, adj, X):
        adj_X = torch.bmm(adj, X)
        adj_X = self.nonlinear_FC(adj_X)

        return adj, adj_X





class prediction(nn.Module):
    def __init__(self, indim_fea, hidden_size):
        super(prediction, self).__init__()
        self.fc1 = nn.Linear(indim_fea, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        features = F.leaky_relu(self.fc1(x))

        features = F.leaky_relu(self.fc2(features))

        return features

class classify(nn.Module):
    def __init__(self, num_class):
        super(classify, self).__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(128, 8)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

        self.cls = nn.Linear(8, num_class)

    def forward(self, features):
        features = self.fc(features)
        cls = self.cls(features)

        return cls
