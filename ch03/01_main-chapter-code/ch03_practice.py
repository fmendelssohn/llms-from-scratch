import torch
from torch import nn
import numpy as np


def softmax(x, dim=-1):
    """Computes softmax of x over dim; output has same shape as x."""
    m = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - m)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, qkv_bias=False) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('inf_mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))
        return

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-1, -2)  # (b, num_tokens, num_tokens)
        attn_scores.masked_fill_(self.inf_mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = softmax(attn_scores / np.sqrt(self.d_out), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, num_tokens, d_out), same as values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_len, dropout=dropout, qkv_bias=qkv_bias)
            for _ in range(num_heads)
        ])
        return
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)  # Concat in last (d_out) dim


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.output_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("inf_mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))
        return

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split last dim of weights 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Swap num_tokens and num_heads dims
        queries = queries.transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-1, -2)  # (b, num_heads, num_tokens, num_tokens) 
        attn_scores.masked_fill_(self.inf_mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = softmax(attn_scores / np.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (
            (attn_weights @ values)  # (b, num_heads, num_tokens, head_dim), same as values
            .transpose(1, 2)         # (b, num_tokens, num_heads, head_dim)
            .contiguous().view(b, num_tokens, self.d_out)  # (num_heads, head_dim) -> d_out
        )
        return self.output_proj(context_vec)  # (b, num_tokens, d_out)
