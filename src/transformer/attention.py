import torch
import torch.nn as nn
import math
from typing import Tuple

def scaled_dot_product_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor, mask:torch.tensor=None) -> Tuple[torch.tensor, torch.tensor]:
    """Famous attention mechanism -> A = softmax(QK^T/sqrt(d))V

    Args:
        Q (torch.tensor): query values
        K (torch.tensor): key values
        V (torch.tensor): value values
        mask (torch.tensor, optional): attention mask. Defaults to None.

    Returns:
        Tuple[torch.tensor, torch.tensor]: resulting attention value and attention scores
    """
    # Find scaling factor - dimensions of the query matrix
    d_k = math.sqrt(Q.size(-1))
    # Shape of Q and K - (batch, heads, seq_len, d_k) - so reshape K to be (batch, heads, d_k, seq_len)
    K_T = K.transpose(-2,-1)
    scores = torch.matmul(Q, K_T) / d_k
    if mask is not None:
        # 0's are ignore, 1's are keep - this is useful if don't allow a current token to pay attention to a future token
        scores = scores.masked_fill(mask == 0, -1e9) # Massive negative number in the 'ignore positions'
    # Softamx
    attn = torch.softmax(scores, dim=-1)
    return (torch.matmul(attn, V), attn)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, num_heads:int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # We need four linear layers
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)
        
    def forward(self, x:torch.tensor, mask:torch.tensor=None) -> torch.tensor:
        """Forward method for the MultiHeadAttention

        Args:
            x (torch.tensor): input tensor
            mask (torch.tensor, optional): Masking tensor. Defaults to None.

        Returns:
            torch.tensor: resulting output
        """
        B, L, C = x.shape # Batch, length, channels/d_model
        
        Q = self.W_q(x)
        Q = Q.reshape((B, L, self.num_heads, self.d_k))
        Q = Q.transpose(1, 2) # B x num_heads x L x d_k - now we can compute attention heads in parallel
        
        # Same for key and value
        K = self.W_k(x)
        K = K.reshape((B, L, self.num_heads, self.d_k))
        K = K.transpose(1, 2)
        
        V = self.W_v(x)
        V = V.reshape((B, L, self.num_heads, self.d_k))
        V = V.transpose(1, 2) # B x num_heads x L x d_k
        
        # Obtain the dot product attention values over all our heads
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        # out - (B, num_heads, L, d_k)
        out = out.transpose(1, 2).contiguous() # back to B x L x num_heads x d_k, and 'contiguous' is a memory layout fix
        out = out.view(B, L, C)    
        out = self.W_o(out)
        return (out, attn)