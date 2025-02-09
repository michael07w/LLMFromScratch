import torch


inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x_1)
     [0.55, 0.87, 0.66], # journey  (x_2)
     [0.57, 0.85, 0.64], # starts   (x_3)
     [0.22, 0.58, 0.33], # with     (x_4)
     [0.77, 0.25, 0.10], # one      (x_5)
     [0.05, 0.80, 0.55]] # step     (x_6)
)

# Softmax should be used for normalization
#   Better at managing extreme values
#   Offers more favorable gradient properties during training
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


# Simplified self-attention mechanism
def simple_self_attention():
    # Calculate attention scores (w) by taking dot product 
    # between query token and all other input tokens
    query = inputs[1]   # 2nd input token will serve as query
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    print('Attention scores for 2nd input token:\n', attn_scores_2)

    # Normalize attention scores to sum to 1
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print('\nAttention weights:', attn_weights_2_tmp)
    print('Sum:', attn_weights_2_tmp.sum())

    # Use "naive" softmax for normalization
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print('\nAttention weights (naive softmax):', attn_weights_2_naive)
    print('Sum:', attn_weights_2_naive.sum())

    # Use PyTorch softmax, optimized for performance
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print('\nAttention weights (softmax):', attn_weights_2)
    print('Sum:', attn_weights_2.sum())

    # Calculate the context vector
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    print('\nContext vector (wrt 2nd input token):\n', context_vec_2)

    # Compute attention scores for all input tokens
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    
    # ^^ can do this faster w/ matrix multiplication
    attn_scores = inputs @ inputs.T
    print('-------------------------------')
    print('\nAttention Scores:\n', attn_scores)

    # Normalize to get attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1) 
    print('\nAttention Weights:\n', attn_weights)


# Self-attention w/ trainable weights
def self_attn_trainable_weights():
    x_2 = inputs[1]

    # input embedding size = 3
    d_in = inputs.shape[1]

    # output embedding size = 2
    d_out = 2

    # Initialize 3 weight matrices
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # Compute query, key, and value vectors for 2nd element
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print('Query (x_2):', query_2)

    # Compute keys and values for all elements
    keys = inputs @ W_key
    values = inputs @ W_value
    print('\nkeys.shape:', keys.shape)
    print('values.shape:', values.shape)

    # Compute attention score for 2nd element
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print('\nAttention score x_2 (q_2 * k_2):', attn_score_22)

    # Compute all attention scores for 2nd element
    attn_scores_2 = query_2 @ keys.T
    print('\nAttention Scores (x_2):\n', attn_scores_2)

    # Normalize the scores to obtain weights
    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax((attn_scores_2 / d_k**0.5), dim=-1)
    print('\nAttention Weights (x_2):\n', attn_weights_2)


if __name__ == '__main__':
    #simple_self_attention()
    self_attn_trainable_weights()