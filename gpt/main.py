import torch

import utils

if __name__ == '__main__':

    ### hyperparameter ###
    utils.set_seed(42)
    block_size = 8 # max_seq_length+1
    batch_size = 4

    ### preprocessing ###
    with open('/data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars) # 65 

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l]) 

    data = torch.tensor(encode(text), dtype=torch.long)

    
