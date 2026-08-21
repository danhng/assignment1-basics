import math
import torch

def softmax(x: torch.Tensor, dim: int, target_indices = None): 
    i_max = torch.max(x, dim=dim, keepdim=True) # max along the dim
    x = x-i_max.values # subtract by max 
    x_e = torch.exp(x)
    i_sum_e = torch.sum(x_e, dim=dim, keepdim=True) # get the sum of e^x
    # select only target indices to calculate and return softmaxes for those indices if provided. 
    if (target_indices is not None):
        # unsqueeze target indices if not matching x.size
        if target_indices.size() != x.size(): 
            target_indices = target_indices[..., None]
        x_e = torch.gather(x_e, dim, index=target_indices)
    return x_e/i_sum_e # return the soft max

def log_softmax(x: torch.Tensor, dim: int, target_indices = None): 
    i_max = torch.max(x, dim=dim, keepdim=True) # max along the dim
    x_shifted = x-i_max.values # subtract by max 
    x_e = torch.exp(x_shifted)
    i_sum_e = torch.sum(x_e, dim=dim, keepdim=True) # get the sum of e^x
    # select only target indices to calculate and return softmaxes for those indices if provided. 
    if (target_indices is not None):
        # unsqueeze target indices if not matching x.size
        if target_indices.size() != x_shifted.size(): 
            target_indices = target_indices[..., None]
        x_shifted = torch.gather(x_shifted, dim, index=target_indices)
    return x_shifted - torch.log(i_sum_e)

'''
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
'''
def scaled_dot_product_attention(q, k, v, mask):
    d_k = q.size(-1)
    pre_softmax = torch.matmul(q, k.mT) / math.sqrt(d_k) # size ..., seq_len, seq_len
    masked = torch.where(mask, 0.0, float('-inf')) #torch.where is very useful when using masked boolean tensors
    pre_softmax = pre_softmax + masked
    softmaxes = softmax(pre_softmax, -1) # size ..., seq_len, seq_len => this is attention score of seq_len * seq_len
    value_weighted_sum = softmaxes.matmul(v) # for a token, we multiply the attention scores of all other tokens with each of the d_k of that token (push all other tokens attention to a single value for each of d_k dim), doing that for all d_k dimensions to form the final representation of the token.
    return value_weighted_sum 

'''
    logits Float[Tensor, "... seq_len vocab_size"]: the logits of predicted tokens, size ..., seq_len, vocab. 
    target: Int[Tensor, "... seq_len"]. The token IDs of the ground truth sequence i+1 in the training set. size ..., seq_len
'''
def cross_entropy(logits, target): 
    #1. compute the loss of the ground truth token (reduce dim -1 from vocab to 1)
    # 1.1. get the logit of the ground truth token
    # 1.2. get the exp of the logit
    # 1.3. get the sum of all logits' exp 
    # 1.4. get the  loss = cross entropy(target_token, predicted prob of target token) = -log(probability(predicted target token)) = -softmax = - (log(exp(ground_truth_token) - log(all logits' exp))
    #2. compute compute the total loss of the entire batch, sequence by averaging the sum of probabilities by D and m. (reduce from ... seq_len to 1)
    cross_entropy_inner_tokens = -log_softmax(logits, dim = -1, target_indices=target)
    cross_entropy_outer = cross_entropy_inner_tokens.mean()
    return cross_entropy_outer

'''
The cosine annealing learning rate schedule takes (i) the current iteration t, (ii) the maximum learningrate lr_max, the minimum (final) learning rate lr_min, 
the number of warm-up iterations T_w, and the final iteration of cosine annealing T_c. 
'''
def cosine_annealing_lr(t, lr_max, lr_min, T_w, T_c): 
    assert T_w <= T_c and t >= 0
    if t < T_w: 
        return t/T_w * lr_max
    if t > T_c: 
        return lr_min
    return lr_min + 1/2*(lr_max-lr_min)*(1+math.cos(math.pi * (t-T_w)/(T_c - T_w)))
    
    