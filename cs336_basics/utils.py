import torch

def softmax(x: torch.Tensor, i: int): 
    i_max = torch.max(x[i])
    tensor_ith = x[i]
    normalized_tensor = tensor_ith - i_max
    sum_logits = torch.sum(torch.pow(normalized_tensor, torch.e))
    softmax = normalized_tensor / sum_logits
    x[i] = softmax
    return x[i]
