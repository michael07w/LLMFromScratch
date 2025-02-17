import torch
import torch.nn as nn


# Considered naive approach
class SelfAttention_v1(nn.Module):
    # Initializes trainable weight matrices (W_query, W_key, W_value)
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # Compute attention scores, normalize to get weights, and multiply by values to get context vectors
    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax((attn_scores / keys.shape[-1]**0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec


# Employs nn.Linear modules for weight matrices
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax((attn_scores / keys.shape[-1]**0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec


# Implements causal mask and dropout components
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)                    
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax((attn_scores / keys.shape[-1]**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec