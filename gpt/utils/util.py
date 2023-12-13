import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np

def set_seed(radom_seed=42):
    torch.manual_seed(radom_seed)
    torch.cuda.manual_seed(radom_seed)
    torch.cuda.manual_seed_all(radom_seed)
    np.random.seed(radom_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(radom_seed)

def get_batch(data, max_seq_length, batch_size): # data loader 만들기
    # generate a small batch of data of inputs x and targets 
    ix = torch.randint(len(data) - max_seq_length, (batch_size,))
    x = torch.stack([data[i:i+max_seq_length] for i in ix])
    y = torch.stack([data[i+1:i+max_seq_length+1] for i in ix])
    return x, y