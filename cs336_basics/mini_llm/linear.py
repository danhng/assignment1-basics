from torch import nn
from torch import Tensor

'''
Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear 
transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear 
module, except for not having a bias argument or parameter. We recommend the following 
interface:
'''
class Linear(nn.Module): 
    def __init__(self, in_features, out_features, device=None, dtype=None): 
        super.__init__()
        self.in_features = in_features # input dimension (x = (sequence_length * in_features) i.e. number of dimensions of embedding input
        self.out_features = out_features # output dimension (W = (out_features, in_features), Y = x * WT = (sequence_length, out_features)) i.e. output hidden layer size
        self.device = device # device to store the parameters
        self.dtype = dtype # Data type of the parameters bf16, fp16 or fp8 etc.

    def forward(self, x: Tensor): 
        pass
