import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

class RandomTextDataset(IterableDataset):
    def __init__(self, data:str, max_seq_length:int):
        super().__init__()
        self.data = data
        self.max_seq_length = max_seq_length
    def __iter__(self) -> Iterator:
        while True:
            i = random.randint(0,len(self.data)-self.max_seq_length-1)
            #i = np.random.randint(0,len(self.data)-self.max_seq_length)
            x = self.data[i:i+self.max_seq_length]
            y = self.data[i+1:i+self.max_seq_length+1]
            yield x, y
    
    # def __len__(self) -> int:
    #     return 1000

# class RandomTextDataset(Dataset):
#     def __init__(self, data:str, max_seq_length:int):
#         super().__init__()
#         self.data = data
#         self.max_seq_length = max_seq_length
#     def __getitem__(self, idx):
#         i = random.randint(0,len(self.data)-self.max_seq_length-1)
#         x = self.data[i:i+self.max_seq_length]
#         y = self.data[i+1:i+self.max_seq_length+1]
#         return x,y