import time
import torch

from utils import set_seed, get_batch
from model import GPT2

def main():
    ### hyperparameter ###
    seed = 42
    max_seq_length = 8 
    batch_size = 4
    num_iter = 10
    learning_rate = 1e-3
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### preprocessing ###
    set_seed(seed)

    with open('/data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars) # 65 

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l]) 

    data = torch.tensor(encode(text), dtype=torch.long)
    
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = GPT2(
        vocab_size=vocab_size,
        emb_dim=512,
        max_seq_length=max_seq_length,
        num_block=8,
        num_head=8,
        ffn_dim=256,
        )
    model = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for _ in range(num_iter):
        x,y = get_batch(train_data,batch_size=batch_size, max_seq_length=max_seq_length)
        x,y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        #optimizer.zero_grad(set_to_none=True)
        #loss.backward()
        #optimizer.step()
    
    # generate from the model
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))


if __name__ == '__main__':
    main()
    