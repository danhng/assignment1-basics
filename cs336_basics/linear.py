import math
from torch import nn
from torch import Tensor
import torch

'''
Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear 
transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear 
module, except for not having a bias argument or parameter. We recommend the following 
interface:
'''
class Linear(nn.Module): 
    def __init__(self, in_features, out_features, device=None, dtype=None, weight=None): 
        super().__init__()
        self.in_features = in_features # input dimension (x = (sequence_length * in_features) i.e. number of dimensions of embedding input
        self.out_features = out_features # output dimension (W = (out_features, in_features), Y = x * WT = (sequence_length, out_features)) i.e. output hidden layer size
        self.device = device # device to store the parameters
        self.dtype = dtype or torch.float32  # Data type of the parameters bf16, fp16 or fp8 etc.
        if weight is not None: 
            weight.to(device)
            weight.to(dtype)
            self.weight = torch.nn.Parameter(weight)
        else: 
            weight = torch.empty(size = (out_features, in_features), dtype=dtype, device=device) # initialize based on Xavier
            std_init_weight = math.sqrt(2/(in_features+out_features))
            nn.init.trunc_normal_(weight, mean=0, std=std_init_weight, a=-3 * std_init_weight, b = 3*std_init_weight)
            self.weight = torch.nn.Parameter(weight)
    
    # x: input tensor of ..., d_model
    def forward(self, x: Tensor): 
        y = torch.matmul(x, self.weight.mT) # y = x * WT
        return y
        
