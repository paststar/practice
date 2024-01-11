'''
Reference : https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master?tab=readme-ov-file
'''

import torch
from torch import nn

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        h : input node embeddings of shape `[n_batches, n_nodes, in_features]`
        adj_mat : adjacency matrix of shape `[n_batches, n_nodes, n_nodes, n_heads]` or `[n_batches, n_nodes, n_nodes, 1]`
        """
        ### after ###
        n_batches, n_nodes, n_emb = h.shape
        g = self.linear(h).view(n_batches, n_nodes, self.n_heads, self.n_hidden) # [n_batches, n_nodes, n_heads, n_hidden]
        g_repeat = g.repeat(1, n_nodes, 1, 1) # [n_batches, n_nodes x n_nodes, n_heads, n_hidden]
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=1) # [n_batches, n_nodes x n_nodes, n_heads, n_hidden]
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1) # [n_batches, n_nodes x n_nodes, n_heads, n_hidden x 2]
        g_concat = g_concat.view(n_batches, n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat)) # [n_batches, n_nodes, n_nodes, n_heads, 1]
        e = e.squeeze(-1) # [n_batches, n_nodes, n_nodes, n_heads]


        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == n_nodes
        assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a) # [n_batches, n_nodes, n_nodes, n_heads]

        attn_res = torch.einsum('bijh,bjhf->bihf', a, g) # [n_batches, n_nodes, n_heads, n_hidden]

        if self.is_concat:
            return attn_res.reshape(n_batches, n_nodes, -1)
        else:
            return attn_res.mean(dim=2)

class GAT(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_batches, n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_batches, n_nodes, n_nodes, n_heads]` or `[n_batches, n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)
