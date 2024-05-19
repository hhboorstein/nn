import torch
from torch import nn

class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, context_len = 4096, p_dropout: float = 0.05):
        """

        Args:
          d_model: embedding dimension
          context_len: maximum length of sequence
          p_dropout: probability for dropout layer
        """

        super().__init__()

        # embedding tensor parts
        pos_idx     = torch.arange(context_len).unsqueeze(1)
        dim_idx     = torch.arange(0, d_model, 2)
        base_change = torch.log(torch.tensor(10_000.0))
        denom       = torch.exp(base_change * dim_idx / d_model)

        # encoding tensor, compatible with input of shape (seq_len, batch_size, emb_dim)
        PE = torch.zeros(context_len, 1, d_model)  
        PE[:, 0, 0::2] = torch.sin(pos_idx / denom)
        PE[:, 0, 1::2] = torch.cos(pos_idx / denom)

        # declare module attributes
        self.register_buffer("PE", PE)
        self.drop = nn.Dropout(p=p_dropout)

    def forward(self, x):
        """

        Args:
          x: Expects a tensor of shape (seq_len, batch_size, emb_dim)

        Returns: encoded tensor of shape (seq_len, batch_size, emb_dim)

        """

        x = x + self.PE[:x.size(0)]
        x = self.drop(x)
        return x