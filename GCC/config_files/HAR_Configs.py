class Config(object):
    def __init__(self):
        # model configs
        # self.input_channels = 9
        self.num_nodes = 9

        self.window_size = 8
        self.time_denpen_len = 16

        self.convo_time_length = 13
        # self.features_len = 18


        self.kernel_size = 3
        self.stride = 1

        self.hidden_channels = 64
        self.final_out_channels = 128
        #
        # self.hidden_channels = 96
        # self.final_out_channels = 64

        self.wavelet_aug = True
        self.random_aug = True


        self.num_classes = 6
        self.dropout = 0.1
        # self.features_len = 18

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128
        self.batch_size_test = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):

        self.max_seg = 3



class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6
