from typing import Literal
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L

from utils import set_seed, get_batch
from model import GPT2, LitGPT
from loader import RandomTextDataset

@dataclass
class Config:
    seed:int
    data_path:str
    max_seq_length:int
    batch_size:int
    num_workers:int
    num_iter:int
    learning_rate:int
    emb_dim:int
    drop_rate:float
    device:Literal['cpu','cuda']

def main(config:Config) -> None:
    ### hyperparameter ###

    ### preprocessing ###
    set_seed(config.seed)

    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read()    
    
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = RandomTextDataset(data[:n],config.max_seq_length)
    val_data = RandomTextDataset(data[n:],config.max_seq_length)

    train_loader = DataLoader(train_data,config.batch_size,num_workers=config.num_workers)
    #val_loader = iter(DataLoader(val_data,config.batch_size,num_workers=0))

    net = GPT2(
        vocab_size=vocab_size,
        emb_dim=config.emb_dim,
        max_seq_length=config.max_seq_length,
        num_block=8,
        num_head=8,
        ffn_dim=config.emb_dim * 4,
        drop_rate=config.drop_rate,
    )

    model = LitGPT(config=config, net=net)
    trainer = L.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader)


    # train_loader = iter(train_loader)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # model = model.to(config.device)
    # print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    
    # for it in tqdm(range(config.num_iter)):
    #     x, y = next(train_loader)
    #     x, y = x.to(config.device), y.to(config.device)
    #     logits, loss = model(x, y)
    #     optimizer.zero_grad(set_to_none=True)
    #     loss.backward()
    #     optimizer.step()

    #     if iter % 10000 == 0 or iter == num_iter - 1:
    #         print(f"step {iter}: train loss {loss:.4f}")
    ### generate from the model ###
    # with torch.no_grad():
    #     context = torch.zeros((1, 1), dtype=torch.long, device=config.device) # use '\n' token
    #     print(decode(model.generate(context, max_gen_len=200)[0].tolist()))


if __name__ == "__main__":
    config = Config(
        seed=42,
        data_path="/data/input.txt",
        max_seq_length=8,
        batch_size=4,
        num_workers=4,
        num_iter=10,
        learning_rate=1e-3,
        emb_dim=128,
        drop_rate=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    main(config=config)