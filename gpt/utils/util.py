import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np

def set_seed(num=42):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(num)