import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttenion(nn.Module):
    def __init__(
            self,
            num_head:int,
            emb_dim:int,
            drop_rate:float,
            ) -> None:
        super().__init__()

        self.num_head = num_head
        #self.emb_dim = emb_dim
        self.wq = nn.Linear(emb_dim,emb_dim)
        self.wk = nn.Linear(emb_dim,emb_dim)
        self.wv = nn.Linear(emb_dim,emb_dim)
        self.norm = (emb_dim/num_head)**(0.5)
        self.linear = nn.Linear(emb_dim,emb_dim)
        self.drop_out = nn.Dropout(drop_rate)

        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_head, -1).permute(0,2,1,3)
        k = self.wk(x).view(batch_size, seq_len, self.num_head, -1).permute(0,2,3,1)
        v = self.wv(x).view(batch_size, seq_len, self.num_head, -1).permute(0,2,1,3)
        score = torch.matmul(q,k)/self.norm

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_head, 1, 1)
            #score.masked_fill_(mask==0, -1e9)
            score.masked_fill_(mask==0, float("-inf"))          
        score = self.drop_out(torch.softmax(score, dim=3)) 

        x = torch.matmul(score,v)
        x = x.transpose(1,2).flatten(start_dim=2) #.reshape(batch_size,seq_len,-1)
        x = self.drop_out(self.linear(x))
        return x
        
class DecoderBlock(nn.Module):
    def __init__(
            self,
            num_head:int,
            emb_dim:int,
            ffn_dim:int,
            drop_rate:float,
            ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)    
        self.msa = MultiHeadAttenion(emb_dim=emb_dim,num_head=num_head,drop_rate=drop_rate)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim,ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim,emb_dim),
            nn.Dropout(drop_rate)
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
            ffn_dim:int,
            drop_rate:float,
            ) -> None:
        super().__init__()
        self.num_block = num_block
        self.max_seq_length = max_seq_length

        self.word_emb = nn.Embedding(vocab_size,emb_dim)
        self.pos_emb = PositionalEmbedding(max_seq_length,emb_dim)
        self.blocks = nn.ModuleList([DecoderBlock(num_head=num_head,emb_dim=emb_dim,ffn_dim=ffn_dim, drop_rate=drop_rate) for _ in range(num_block)])
        self.out = nn.Linear(emb_dim, vocab_size)
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_length,max_seq_length)).cuda())

    def forward(self, x, y=None):
        x = self.word_emb(x)+self.pos_emb(x)
        for block in self.blocks:
            x = block(x,self.mask)
        logits = self.out(x) # add layernorm?
        if y is None:
            loss = None
        else:
            b,s,e = logits.shape
            logits = logits.view(-1,e)
            y = y.flatten()
            loss = F.cross_entropy(logits,y)
        return logits, loss

    def generate(self, x, max_gen_len):
        for _ in range(max_gen_len):
            x = x if x.size(1) <=self.max_seq_length else x[:,-self.max_seq_length:]
            logits,_ = self(x) # we can use max_seq_length token or less
            logits = logits[:,-1,:] # use last token
            probs = F.softmax(logits,dim=-1)
            x = torch.cat((x,torch.multinomial(probs, 1)), dim=1)
        return x
    
if __name__ == '__main__':
    pass