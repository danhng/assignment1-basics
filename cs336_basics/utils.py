import math
import torch

def softmax(x: torch.Tensor, i: int): 
    i_max = torch.max(x, dim=i, keepdim=True) # max along the dim
    x = x-i_max.values # subtract by max 
    i_sum_e = torch.sum(torch.exp(x), dim=i, keepdim=True) # get the sum of e^x
    return torch.exp(x)/i_sum_e # return the soft max

x = torch.Tensor([[1,2], [3, 5]])
i_max = torch.max(x, dim=-2, keepdim=True) # max along the dim
x = x-i_max.values
i_sum_e = torch.sum(torch.exp(x), dim=-1, keepdim=True)
print(torch.exp(x)/i_sum_e)


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
    