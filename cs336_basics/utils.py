import math
import torch

def softmax(x: torch.Tensor, dim: int): 
    i_max = torch.max(x, dim=dim, keepdim=True) # max along the dim
    x = x-i_max.values # subtract by max 
    i_sum_e = torch.sum(torch.exp(x), dim=dim, keepdim=True) # get the sum of e^x
    return torch.exp(x)/i_sum_e # return the soft max

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
    #1. compute softmaxes of each token position in each sequence (reduce last dimen from vocab to 1) based on target. 
    #2. compute the probability of each token position based on softmax (transform from softmax to probability)
    #3. compute compute the total loss of the entire batch, sequence by averaging the sum of probabilities by D and m. (reduce from ... seq_len to 1)
    pass