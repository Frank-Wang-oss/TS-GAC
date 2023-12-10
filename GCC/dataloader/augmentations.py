import numpy as np
import torch
import pywt
import matplotlib.pyplot as plt



def changes_correlations(Adj, num_remained):


    _, idx = torch.sort(Adj, descending=True, dim=-1)
    zero = torch.zeros_like(Adj)

    topk = idx[:, :, :num_remained]

    bat_id = torch.arange(Adj.size(0)).unsqueeze(1).unsqueeze(1)
    sensor_id = torch.arange(Adj.size(1)).unsqueeze(1).unsqueeze(0)
    zero[bat_id, sensor_id, topk] = 1

    rand_ = torch.rand(Adj.size())
    rand_[bat_id,sensor_id, topk] = 0

    rand_ = rand_.cuda() if torch.cuda.is_available() else rand_
    coeffi = zero + rand_

    return Adj * coeffi


def wavelet_transform(x, weak=True):


    b_g, b_d = pywt.dwt(x, 'db2')
    if weak:
        b_d = b_d + np.random.random(b_d.shape) * 0.1
    else:
        b_g = b_g + np.random.random(b_g.shape) * 0.1

    a_ = pywt.idwt(b_g, b_d, 'db2')


    return a_


def partial_changes_MTS(x, coeffi, num_maintain):
    ### x size is (bs, N, time_length)
    x_aug = wavelet_transform(x, coeffi)

    bs, N, time_length = x.shape

    idx = np.random.randint(low=0, high=N, size=[bs,num_maintain])
    bat_id = np.expand_dims(np.arange(bs),1)

    x[bat_id,idx] = x_aug[bat_id,idx]

    return x
def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    strong_aug = scaling(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_scale_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

