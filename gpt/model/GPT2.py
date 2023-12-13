#from typing import 

import torch
import torch.nn as nn
import torch.nn.functional as F

# Q) GPT는 train할 때 input으로 몇개의 token을 넣나? 1개만 쓰나?
# einsum ver 추가
# ML test 코드

class MultiHeadAttenion(nn.Module):
    def __init__(
            self,
            num_head:int,
            emb_dim:int,
            ) -> None:
        super().__init__()

        self.num_head = num_head
        #self.emb_dim = emb_dim
        self.wq = nn.Linear(emb_dim,emb_dim)
        self.wk = nn.Linear(emb_dim,emb_dim)
        self.wv = nn.Linear(emb_dim,emb_dim)
        self.denom = (emb_dim/num_head)**(0.5)
        self.mlp = nn.Linear(emb_dim,emb_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_head, -1).permute(0,2,1,3)
        k = self.wk(x).view(batch_size, seq_len, self.num_head, -1).permute(0,2,3,1)
        v = self.wv(x).view(batch_size, seq_len, self.num_head, -1).permute(0,2,1,3)
        score = torch.matmul(q,k)/self.denom

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
            #score.masked_fill_(mask==0, -1e9)
            score.masked_fill_(mask==0, float("-inf"))          
        score = torch.softmax(score, dim=3)        

        x = torch.matmul(score,v)
        x = x.transpose(1,2).flatten(start_dim=2) #.reshape(batch_size,seq_len,-1)
        x = self.mlp(x)
        return x
        
class DecoderBlock(nn.Module):
    def __init__(
            self,
            num_head:int,
            emb_dim:int,
            ffn_dim:int,
            ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)    
        self.msa = MultiHeadAttenion(emb_dim=emb_dim,num_head=num_head)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim,ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim,emb_dim)
        )

    def forward(self, x, mask=None):
        x = self.ln1(x)
        x = self.ln2(self.msa(x, mask)+x)
        x = self.ffn(x) + x
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, emd_dim):
        super().__init__()
        self.pe = nn.Embedding(max_seq_length, emd_dim)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class GPT2(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            max_seq_length:int,
            emb_dim:int,
            num_block:int,
            num_head:int,
            ffn_dim:int
            ) -> None:
        super().__init__()

        self.word_emb = nn.Embedding(vocab_size,emb_dim)
        self.pos_emb = PositionalEmbedding(max_seq_length,emb_dim)
        self.blokcs = nn.Sequential(*[DecoderBlock(num_head=num_head,emb_dim=emb_dim,ffn_dim=ffn_dim) for _ in range(num_block)])
        self.out = nn.Linear(emb_dim, vocab_size)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.word_emb(x)+self.pos_emb(x)
        x = self.blokcs(x)
        x = self.out(x)
        loss = self.loss_func(x,y)
        return x,loss

    def generate(x):
        pass


if __name__ == '__main__':
    batch_size, num_head, seq_len, emb_dim = 5, 4, 7, 12
    x = torch.randn((batch_size,seq_len,emb_dim))
    mask = torch.randint(0,2,(batch_size,seq_len,seq_len))
    
    #tmp = GPT2(num_head=num_head,emb_dim=emb_dim,ffn_dim=emb_dim//2)
    # tmp = nn.Sequential(
    #     DecoderBlock(num_head=num_head,emb_dim=emb_dim,ffn_dim=emb_dim//2),
    #     DecoderBlock(num_head=num_head,emb_dim=emb_dim,ffn_dim=emb_dim//2)
    #     )
    tmp = DecoderBlock(num_head=num_head,emb_dim=emb_dim,ffn_dim=emb_dim//2)

    print(tmp(x))

    # inf = float('-inf')
    # x = torch.Tensor([1,2,3,4,5,6,7,8,9,10]) 
    # mask = torch.Tensor([1,1,0,0,1,0,0,0,0,0])
    # x.masked_fill_(mask==0, inf)
    # print(torch.softmax(x,dim=0))
    