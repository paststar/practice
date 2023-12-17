from typing import Literal
from dataclasses import dataclass
import time

import torch
from tqdm import tqdm

from utils import set_seed, get_batch
from model import GPT2


@dataclass
class Config:
    seed: int
    max_seq_length:int
    batch_size: int
    num_iter: int
    learning_rate: int
    emb_dim: int
    drop_rate: float
    device:Literal['cpu','cuda']


def main(config):
    ### hyperparameter ###

    ### preprocessing ###
    set_seed(config.seed)

    with open("/data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)  # 65

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = GPT2(
        vocab_size=vocab_size,
        emb_dim=config.emb_dim,
        max_seq_length=config.max_seq_length,
        num_block=8,
        num_head=8,
        ffn_dim=config.emb_dim * 4,
        drop_rate=config.drop_rate,
    )
    model = model.to(config.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in tqdm(range(config.num_iter)):
        x, y = get_batch(
            train_data, batch_size=config.batch_size, max_seq_length=config.max_seq_length
        )
        x, y = x.to(config.device), y.to(config.device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        break

        if iter % 10000 == 0 or iter == num_iter - 1:
            print(f"step {iter}: train loss {loss:.4f}")

    # generate from the model
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        print(decode(model.generate(x, max_gen_len=200)[0].tolist()))


if __name__ == "__main__":
    config = Config(
        seed=42,
        max_seq_length=8,
        batch_size=4,
        num_iter=10,
        learning_rate=1e-3,
        emb_dim=128,
        drop_rate=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    main(config=config)