from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule

class LitGPT(LightningModule):
    def __init__(self, config, net:torch.nn.Module) -> None:
        super().__init__()
        self.config = config
        self.net = net
    
    def forward(self, x:torch.Tensor, max_gen_len:int) -> Any:
        return self.net.generate(x,max_gen_len)
    
    def training_step(self, batch, batch_idx) -> Any:
        x,y = batch
        _, loss = self.net(x,y)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.net.parameters(), lr=self.config.learning_rate)