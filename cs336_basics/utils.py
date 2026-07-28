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

