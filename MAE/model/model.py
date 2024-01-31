import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from base import BaseModel

'''
### Note ###
1. encoder에선 random masking 적용 후 unmaksing된 patch만 사용하고, decoer에선 mask token + unmask token(by encoder) 사용함
 => random masking 이후 순서 복원 필요
2. encoder/decoder 모두 sin-cos positional encoding 사용
3. finetunning시 사용을 위해 pretain에서 dummy class token 사용 (class token 없이 global pooling 해도 됨)
4. encoder, decoder width가 달라 중간에 linaer projection이 들어감
5. patch 단위의 normalized pixel에 대한 MSE를 loss로 사용
6. fine-tunning시 drop path, cutmix, mixup, label smoothing 사용
'''

class MAE(BaseModel):
    def __init__(self, image_size, patch_size, mask_ratio, encoder_type, dropout):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_feature = patch_size**2*3
        self.seq_len = (image_size//patch_size)**2
        self.unmask_seq_len = int(self.seq_len*(1-mask_ratio))
        self.eps = 1e-6

        if encoder_type == 'ViT-S':
            encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6, dim_feedforward=1536, dropout=dropout, activation='gelu',batch_first=True, norm_first=True)
            enc_num_layer = 6
            self.enc_dim = 384
        elif encoder_type == 'ViT-B':
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=dropout, activation='gelu',batch_first=True, norm_first=True)
            enc_num_layer = 12
            self.enc_dim = 768
        elif encoder_type == 'ViT-L':
            encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=4096, dropout=dropout, activation='gelu',batch_first=True, norm_first=True)
            enc_num_layer = 24
            self.enc_dim = 1024
        
        self.patch_emb = nn.Linear(self.num_feature, self.enc_dim)
        self.register_pe('enc_pe', self.unmask_seq_len, self.enc_dim)
        self.cls_token = nn.Parameter(torch.randn(self.enc_dim))
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=enc_num_layer)

        self.dec_dim = 512
        self.register_pe('dec_pe', self.seq_len, self.dec_dim)
        self.mask_token = nn.Parameter(torch.randn(self.dec_dim))
        self.proj = nn.Linear(self.enc_dim, self.dec_dim) # projection
        decoder_layer = nn.TransformerEncoderLayer(d_model=self.dec_dim, nhead=16, dim_feedforward=2048, dropout=dropout, activation='gelu',batch_first=True, norm_first=True)
        self.decoder = nn.Sequential(
            nn.TransformerEncoder(encoder_layer=decoder_layer, num_layers=8),
            nn.Linear(self.dec_dim, self.num_feature) # patch reconstruction
        )
        
    def patchify(self, x):
        '''
        x : images (B,C,H,W) 
        return : seq of patches (B,np*np, C*ps*ps) where np = #patch and ps = patch size 
        '''
        return rearrange(x,"b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
        
    def register_pe(self, buf_name:str, seq_len:int, emb_dim:int) -> None:
        '''
        buf_name(str) : buffer name
        seq_len(int) : sequnce length 
        emb_dim : embedding dimension
        '''
        assert emb_dim % 2 == 0 # Can't be used unless emb_dim is even
        exp = torch.arange(0, emb_dim, 2)/emb_dim
        x = torch.arange(seq_len).unsqueeze(1)*torch.pow(10000,-exp)
        pe = torch.zeros(seq_len, emb_dim)
        pe[:,0::2] = torch.sin(x)
        pe[:,1::2] = torch.cos(x)
        self.register_buffer(buf_name, pe)

    
    def random_masking(self, x:torch.Tensor):
        '''
        x : seq of patches (batch size, length, feature)
        return : unmask (batch size, unmask length, feature) and unmask index (batch size, unmask length)
        '''
        B, L, E = x.shape
        # num_unmask = int(L*(1-self.mask_ratio))
        random_ind = torch.argsort(torch.randn(B,L,device=x.device),dim=1)

        unmask_ind = random_ind[:,:self.unmask_seq_len]
        x_unmask = torch.gather(x,1,unmask_ind.unsqueeze(-1).expand(-1,-1,E))
        return x_unmask, unmask_ind
    
    def pretrain_loss(self, x:torch.Tensor):
        '''
        x : images (batch size, channel, height, width)
        '''

        x = self.patchify(x)
        B, L, _ = x.shape
        target = (x - torch.mean(x, dim=2, keepdim=True))/(torch.std(x, dim=2, keepdim=True) + self.eps) # normalized per patch

        ### encoding ###
        x_unmask, unmask_ind = self.random_masking(x)
        x_unmask = self.patch_emb(x_unmask)
        x_unmask += self.enc_pe
        x_unmask = torch.cat([self.cls_token.expand(B,1,-1),x_unmask],dim=1) # add dummy cls token
        x_unmask = self.encoder(x_unmask)
        
        ### decoding ###
        x_unmask = self.proj(x_unmask)
        x_unmask = x_unmask[:,1:,:] # remove cls token
        x_full = torch.scatter(self.mask_token.expand(B,L,-1), 1, unmask_ind.unsqueeze(-1).expand(-1,-1,self.dec_dim), x_unmask) 
        x_full += self.dec_pe
        x_full = self.decoder(x_full)
        
        ### loss ###
        mask = torch.ones((B,L), dtype=torch.bool, device=x_full.device)
        mask.scatter_(1,unmask_ind, False)
        loss = torch.mean((target-x_full)**2,dim=2)
        loss = loss*mask # use only mask token
        return loss.sum()/mask.sum()

    def encoding(self, x:torch.Tensor):
        x = self.patchify(x)
        x = self.patch_emb(x)
        x += self.enc_pe
        x = torch.cat([self.cls_token.expand(B,1,-1),x],dim=1) # add cls token
        x = self.encoder(x)

if __name__ == '__main__':
    from typing import Literal
    from dataclasses import dataclass

    @dataclass
    class Config:
        # seed:int
        device:Literal['cpu','cuda']
        # data_path:str
        # val_ratio:float
        # batch_size:int
        # num_workers:int
        # num_epoch:int
        # learning_rate:int
        image_size:int
        patch_size:int
        mask_ratio:float
        encoder:Literal['ViT-S','ViT-B','ViT-L']
        dropout:float

    config = Config(
        device='cpu', # "cuda" if torch.cuda.is_available() else "cpu",
        image_size=224,
        patch_size=16,
        mask_ratio=0.75,
        encoder='ViT-S',
        dropout=0.5,
    )

    B,C,H,W = 8,3,224,224
    x = torch.rand(B,C,H,W).to(config.device)
    model = MAE(config).to(config.device)
    loss = model.pretrain_loss(x)
    loss.backward()
    